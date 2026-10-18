"""Run `at-run` on a Modal H200; every checkpoint is kept on a Modal Volume.

    uvx modal run --detach modal_app.py --job jobs/example_stack_sponges.json \
        --args "--n 1000 --steps 30000 --save-freq 5000 --workers 16"

What happens:
  1. Locally: the job file, its clips and its setup folder are uploaded to the `adapter-training-work`
     volume (same relative paths as in this repo).
  2. On Modal (H200, 16 CPUs, 128 GB RAM, up to 24 h): SAM 3D Objects checkpoints are downloaded to the
     `adapter-training-cache` volume on the first run, then `at-run` runs with the job.
  3. Outputs go to the work volume: /outputs/train/<skill>/vN/checkpoints/* (every checkpoint),
     /outputs/skills/<skill>/ (report.json, sim_eval/, videos), /lerobot/<skill> (dataset). The volume
     is committed every 10 minutes while it runs, so checkpoints survive a crash or timeout.
     Fetch them with `--download` or `modal volume get adapter-training-work outputs/train/<skill> .`

Optional: Google Drive. If the secret also has RCLONE_CONFIG_GDRIVE_TYPE=drive,
RCLONE_CONFIG_GDRIVE_SCOPE=drive and RCLONE_CONFIG_GDRIVE_TOKEN=<JSON from `rclone authorize drive`>,
checkpoints and reports are also copied to Drive (--drive-folder) every 10 minutes and at the end.

One-time setup (QUICKSTART.md, "Run on Modal"): a Modal secret named `adapter-training` with HF_TOKEN.
Optional keys: ANTHROPIC_API_KEY, WANDB_API_KEY, the RCLONE_CONFIG_GDRIVE_* values above.

Locally this file only needs `modal` (run it with uvx as above); everything else is in the image.
"""

import json
import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent
REMOTE = "/root/AdapterTraining"  # PROJECT_ROOT inside the container
VENV_PY = f"{REMOTE}/.venv/bin/python"
SAM3D_REPO = "/opt/sam-3d-objects"
SAM3D_PYTHON = "/opt/conda/envs/sam3d-objects/bin/python"
GPU = "H200"
DRIVE_REMOTE = "gdrive"  # rclone remote, configured by the RCLONE_CONFIG_GDRIVE_* secret values

IGNORE = ["**/__pycache__", "**/*.pyc"]

image = (
    # CUDA 12.1 devel: SAM 3D Objects' environment is built for cu121 and compiles CUDA extensions.
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git", "curl", "ca-certificates", "build-essential", "unzip",
        "ffmpeg",  # FFmpeg libraries for torchcodec (LeRobot's video decoding)
        "libgl1", "libegl1", "libglvnd0", "libosmesa6", "libglib2.0-0",  # MuJoCo / OpenCV rendering
    )
    # NVIDIA's EGL vendor file, so MuJoCo can render headless on the GPU (OSMesa is the fallback).
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d && echo "
        """'{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' """
        "> /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "curl -fsSL https://rclone.org/install.sh | bash",
    )
    # SAM 3D Objects in its own conda environment (heavy layer: rebuilt only if the script changes).
    # 1) conda + the environment from its default.yml (kept inline so editing the script below
    #    doesn't redo this ~10 minute step)
    .run_commands(
        "curl -fsSL -o /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/"
        "Miniforge3-$(uname)-$(uname -m).sh && bash /tmp/Miniforge3.sh -b -p /opt/conda && rm /tmp/Miniforge3.sh",
        f"git clone https://github.com/facebookresearch/sam-3d-objects.git {SAM3D_REPO}",
        f"/opt/conda/bin/mamba env create -y -f {SAM3D_REPO}/environments/default.yml && /opt/conda/bin/conda clean -afy",
    )
    # 2) its pip packages and CUDA extensions
    .add_local_file(LOCAL / "scripts" / "modal_image" / "install_sam3d.sh", "/opt/install_sam3d.sh", copy=True)
    .run_commands("sed -i 's/\\r$//' /opt/install_sam3d.sh && bash /opt/install_sam3d.sh code")
    .run_commands("bash /opt/install_sam3d.sh packages")
    .run_commands("bash /opt/install_sam3d.sh finish")
    # One of SAM 3D's dependencies writes pypi.ngc.nvidia.com (no longer exists) into a global pip
    # config, which makes every later pip install retry it. Remove it.
    .run_commands(
        "for f in /etc/pip.conf /etc/xdg/pip/pip.conf /root/.pip/pip.conf /root/.config/pip/pip.conf "
        "/opt/conda/envs/sam3d-objects/pip.conf; do [ -f $f ] && grep -l ngc.nvidia $f && rm -f $f; done; "
        "pip config list -v || true"
    )
    # This repo's uv environment from the lock file (rebuilt only if pyproject/uv.lock change).
    .pip_install("uv")
    .add_local_file(LOCAL / "pyproject.toml", f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file(LOCAL / "uv.lock", f"{REMOTE}/uv.lock", copy=True)
    .add_local_file(LOCAL / "README.md", f"{REMOTE}/README.md", copy=True)
    .add_local_dir(LOCAL / "plugins", f"{REMOTE}/plugins", copy=True, ignore=IGNORE)
    .run_commands(f"cd {REMOTE} && uv sync --frozen --no-install-project")
    # WiLoR hand tracker, keeping the locked torch (see the script).
    .add_local_file(LOCAL / "scripts" / "install_wilor.sh", f"{REMOTE}/scripts/install_wilor.sh", copy=True)
    .run_commands(f"sed -i 's/\\r$//' {REMOTE}/scripts/install_wilor.sh && NO_SYNC=1 bash {REMOTE}/scripts/install_wilor.sh")
    .env({
        "PYTHONPATH": f"{REMOTE}/src",  # the project itself isn't installed into the venv
        "SAM3D_REPO": SAM3D_REPO,
        "SAM3D_PYTHON": SAM3D_PYTHON,
        "AT_OUTPUTS_DIR": "/work/outputs",
        "AT_LEROBOT_DIR": "/work/lerobot",
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
    })
    # The code itself (cheap layers: a code change only rebuilds from here).
    .add_local_dir(LOCAL / "src", f"{REMOTE}/src", copy=True, ignore=IGNORE)
    .add_local_dir(LOCAL / "scripts", f"{REMOTE}/scripts", copy=True, ignore=IGNORE)
    .run_commands(
        # OpenArm URDF -> MuJoCo model, and the MediaPipe hand model (data/ inside the image).
        f"cd {REMOTE} && {VENV_PY} -m adapter_training.video.robot",
        f"cd {REMOTE} && {VENV_PY} -c \"import urllib.request; from adapter_training.common import DATA_DIR; "
        "from adapter_training.video.hands import MEDIAPIPE_MODEL_URL as u; "
        "(DATA_DIR / 'models').mkdir(parents=True, exist_ok=True); "
        "urllib.request.urlretrieve(u, DATA_DIR / 'models' / 'hand_landmarker.task')\"",
        # Import check of everything the pipeline needs (no GPU at build time).
        f"cd {REMOTE} && {VENV_PY} -c \"import torch, mujoco, mink, coacd, xatlas, lerobot, torchcodec, mediapipe, "
        "lerobot_robot_openarm_overlay; "
        "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; "
        "import adapter_training.run, adapter_training.sim_eval; print('AdapterTraining env OK: torch', torch.__version__)\"",
        # Build WiLoR once on the CPU: bakes its weights into the image and checks its MANO model loads.
        f"cd {REMOTE} && {VENV_PY} -c \"from adapter_training.video.hands import WiLoRTracker; "
        "WiLoRTracker(device='cpu'); print('WiLoR weights OK')\"",
    )
)

app = modal.App("adapter-training", image=image)
# Model caches on the cache volume. Set at run time only: build steps must not write into /cache
# (a volume can't be mounted over a non-empty folder).
RUNTIME_ENV = {"HF_HOME": "/cache/huggingface", "TORCH_HOME": "/cache/torch"}
cache = modal.Volume.from_name("adapter-training-cache", create_if_missing=True)
work = modal.Volume.from_name("adapter-training-work", create_if_missing=True)
# AT_MODAL_NO_SECRET=1: run without the secret (the `check`, or test jobs that need no gated models).
secrets = [] if os.environ.get("AT_MODAL_NO_SECRET") else [modal.Secret.from_name("adapter-training")]


def _ensure_sam3d_checkpoints() -> None:
    target = Path("/cache/sam3d/hf")
    if (target / "pipeline.yaml").exists():
        return
    print("[modal] downloading SAM 3D Objects checkpoints (first run only)")
    tmp = Path("/cache/sam3d/hf-download")
    subprocess.run([VENV_PY, "-c", "import sys; from huggingface_hub import snapshot_download; "
                    "snapshot_download('facebook/sam-3d-objects', local_dir=sys.argv[1], max_workers=4)", str(tmp)],
                   check=True)
    (tmp / "checkpoints").rename(target)
    subprocess.run(["rm", "-rf", str(tmp)], check=True)
    cache.commit()


def _pick_gl_backend() -> dict:
    """EGL (GPU) if MuJoCo can render with it here, else OSMesa (CPU, slower but always works)."""
    probe = "import mujoco; mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 8, 8)"
    for env in ({"MUJOCO_GL": "egl"}, {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}):
        if subprocess.run([VENV_PY, "-c", probe], env={**os.environ, **env}, capture_output=True).returncode == 0:
            print(f"[modal] MuJoCo rendering: {env['MUJOCO_GL']}")
            return env
    raise RuntimeError("MuJoCo can render with neither EGL nor OSMesa in this container")


def _upload_to_drive(folder: str) -> None:
    """Copy every checkpoint (outputs/train/**) and the skill reports to Google Drive. Incremental."""
    cmd = ["rclone", "copy", "/work/outputs", f"{DRIVE_REMOTE}:{folder}", "--skip-links", "--transfers", "8",
           "--include", "train/**", "--include", "skills/*/report.json", "--include", "skills/*/sim_eval/**"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[modal] Drive upload failed (will retry): {result.stderr[-1000:]}")
    else:
        print(f"[modal] uploaded checkpoints to Drive: {folder}/")


@app.function(gpu=GPU, cpu=16, memory=128 * 1024, timeout=24 * 3600, secrets=secrets, env=RUNTIME_ENV,
              volumes={"/cache": cache, "/work": work})
def run_job(job_rel: str, extra: list[str], drive_folder: str, upload_every_s: int = 600) -> dict:
    job = json.loads((Path("/work/inputs") / job_rel).read_text())
    reconstructor = next((extra[i + 1] for i, a in enumerate(extra[:-1]) if a == "--reconstructor"),
                         job.get("reconstructor", "sam3d"))
    if reconstructor == "sam3d":
        _ensure_sam3d_checkpoints()
    env = {**os.environ, **_pick_gl_backend()}
    drive = bool(os.environ.get(f"RCLONE_CONFIG_{DRIVE_REMOTE.upper()}_TYPE"))
    if not drive:
        print("[modal] outputs are kept on the volume adapter-training-work (no Google Drive configured)")

    stop = threading.Event()

    def sync_loop() -> None:  # checkpoints reach Drive while training runs, not only at the end
        while not stop.wait(upload_every_s):
            work.commit()
            if drive:
                _upload_to_drive(drive_folder)

    syncer = threading.Thread(target=sync_loop, daemon=True)
    syncer.start()
    try:
        cmd = [VENV_PY, "-m", "adapter_training.run", "--job", job_rel, *extra]
        print("[modal] " + shlex.join(cmd))
        code = subprocess.run(cmd, cwd="/work/inputs", env=env).returncode
    finally:
        stop.set()
        syncer.join()
        work.commit()
        if drive:
            _upload_to_drive(drive_folder)

    name = next((extra[i + 1] for i, a in enumerate(extra[:-1]) if a == "--name"), job.get("name", ""))
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")  # as perception.perceive.slug
    report = Path("/work/outputs/skills") / slug / "report.json"
    return {"exit_code": code, "skill": slug, "report": json.loads(report.read_text()) if report.exists() else None}


@app.function(gpu=GPU, timeout=1800, env=RUNTIME_ENV, volumes={"/cache": cache})
def check() -> None:
    """`AT_MODAL_NO_SECRET=1 modal run modal_app.py::check`: build the image and test it on the GPU."""
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], check=True)
    _pick_gl_backend()
    subprocess.run([VENV_PY, "-c", "import torch; x = torch.randn(1024, 1024, device='cuda'); "
                    "print('uv env:', torch.__version__, torch.cuda.get_device_name(), float((x @ x).sum()) == float((x @ x).sum()))"],
                   check=True)
    subprocess.run([SAM3D_PYTHON, "-c", "import torch, pytorch3d, kaolin, gsplat; from pytorch3d.ops import knn_points; "
                    "p = torch.randn(1, 64, 3, device='cuda'); knn_points(p, p, K=2); "
                    "print('SAM 3D env:', torch.__version__, torch.cuda.get_device_name(), 'pytorch3d CUDA ops OK')"],
                   check=True)
    subprocess.run([VENV_PY, "-m", "adapter_training.perception.reconstruct"])  # repo found once checkpoints exist
    print("check passed")


def _job_files(job_path: Path, extra: list[str]) -> list[Path]:
    job = json.loads(job_path.read_text())
    setup = Path(next((extra[i + 1] for i, a in enumerate(extra[:-1]) if a == "--setup"),
                      job.get("setup", "data/setup")))
    files = [job_path, *map(Path, job["videos"])]
    if not setup.is_dir():
        raise SystemExit(f"setup folder {setup} not found (needs at least tag.mp4; see QUICKSTART.md section 2)")
    files += [p for p in setup.iterdir() if p.is_file()]
    # Any other local path the job refers to, e.g. "masks:<dir>" / "mesh:<file>" (tests, known objects).
    for value in [*map(str, job.values()), *extra]:
        candidate = Path(value.split(":", 1)[1] if value.startswith(("masks:", "mesh:")) else value)
        if value and candidate.exists() and candidate.resolve().is_relative_to(LOCAL) and candidate != Path("."):
            files += [candidate] if candidate.is_file() else [p for p in candidate.rglob("*") if p.is_file()]
    files = list(dict.fromkeys(files))
    for f in files:
        if not f.exists():
            raise SystemExit(f"{f} not found")
        if not f.resolve().is_relative_to(LOCAL):
            raise SystemExit(f"{f} is outside {LOCAL}: put clips and setup files inside the repo")
    return files


def _download(remote_dir: str, local_dir: Path) -> int:
    """Copy a folder from the work volume (e.g. outputs/train/<skill>) into this repo."""
    count = 0
    for entry in work.listdir(remote_dir, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        target = local_dir / Path(entry.path).relative_to(remote_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            for chunk in work.read_file(entry.path):
                fh.write(chunk)
        count += 1
    return count


@app.local_entrypoint()
def main(job: str, args: str = "", download: bool = False, drive_folder: str = "AdapterTraining") -> None:
    """--job jobs/x.json  --args "<extra at-run options>"  [--download]  [--drive-folder <Drive folder>]"""
    os.chdir(LOCAL)  # job paths are relative to the repo, as for `uv run at-run`
    extra = shlex.split(args)
    job_path = Path(job)
    files = _job_files(job_path, extra)
    with work.batch_upload(force=True) as batch:
        for f in files:
            rel = f.resolve().relative_to(LOCAL).as_posix()  # same layout on the volume
            batch.put_file(f, f"/inputs/{rel}")
    print(f"Uploaded {len(files)} files to volume adapter-training-work:/inputs")

    result = run_job.remote(job_path.resolve().relative_to(LOCAL).as_posix(), extra, drive_folder)
    report = result["report"] or {}
    print(f"\nexit code {result['exit_code']}: {report.get('result', {})}")
    if "sim_evaluation" in report.get("stages", {}):
        se = report["stages"]["sim_evaluation"]
        print(f"simulated success {se.get('new_success_rate')}; checkpoints {se.get('checkpoints')}")
    name = result["skill"]
    print(f"Checkpoints: volume adapter-training-work, /outputs/train/{name}/")
    if download and "training" in report.get("stages", {}):
        n = _download(f"outputs/train/{name}", LOCAL / "outputs" / "train" / name)
        _download(f"outputs/skills/{name}/sim_eval", LOCAL / "outputs" / "skills" / name / "sim_eval")
        print(f"Downloaded {n} checkpoint files to outputs/train/{name}/")
