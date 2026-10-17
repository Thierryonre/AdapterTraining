"""DexMimicGen-style data generation, reimplemented on the MuJoCo scene env.

For each new episode:
  1. pick a source demo and move every object to a random pose near where it was in that demo
  2. for each phase, take the source's commanded gripper poses and re-express them relative to where
     the phase's reference object is *now* (arms sharing a reference share one transform), or replay
     them as recorded (no reference)
  3. interpolate from the current gripper poses to the start of each transformed segment
  4. execute through IK + physics, keep the episode only if the task's success check passes
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from adapter_training.datagen.dataset import Episode, Recorder, SourceDemo
from adapter_training.datagen.env import SIDES, SceneEnv, pose_matrix, yaw_pose
from adapter_training.datagen.task import Phase


@dataclass
class GenConfig:
    # Object pose randomisation, relative to the chosen source demo's initial object poses.
    dx: tuple[float, float] = (-0.04, 0.04)
    dy: tuple[float, float] = (-0.05, 0.05)
    dyaw_deg: tuple[float, float] = (-20.0, 20.0)
    interp_steps: int = 8  # steps to move from the current pose to each transformed segment
    hold_steps: int = 10  # steps to hold at the end before checking success
    action_noise_m: float = 0.0  # optional position noise on commanded poses (MimicGen uses ~5 mm)
    min_gap_m: float = 0.01  # objects' footprints must be at least this far apart


def _footprint_radius(size) -> float:
    return float(np.hypot(size[0], size[1]) / 2)


def sample_object_poses(src: SourceDemo, env: SceneEnv, cfg: GenConfig, rng, tries: int = 50) -> dict[str, np.ndarray]:
    """Random pose per object around its source pose, upright on the table, footprints not overlapping."""
    for _ in range(tries):
        poses = {}
        for name in env.object_names:
            base = src.object_pose[name][0]
            yaw = np.arctan2(base[1, 0], base[0, 0]) + np.radians(rng.uniform(*cfg.dyaw_deg))
            x = base[0, 3] + rng.uniform(*cfg.dx)
            y = base[1, 3] + rng.uniform(*cfg.dy)
            poses[name] = yaw_pose(x, y, env.object_size(name)[2] / 2, yaw)
        names = list(poses)
        clear = all(
            np.linalg.norm(poses[a][:2, 3] - poses[b][:2, 3])
            > _footprint_radius(env.object_size(a)) + _footprint_radius(env.object_size(b)) + cfg.min_gap_m
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        )
        if clear:
            return poses
    return {n: src.object_pose[n][0] for n in env.object_names}  # fall back to the source layout


# Backwards-compatible single-object helper.
def sample_object_pose(base: np.ndarray, cfg: GenConfig, rng: np.random.Generator, height: float) -> np.ndarray:
    yaw = np.arctan2(base[1, 0], base[0, 0]) + np.radians(rng.uniform(*cfg.dyaw_deg))
    return yaw_pose(base[0, 3] + rng.uniform(*cfg.dx), base[1, 3] + rng.uniform(*cfg.dy), height / 2, yaw)


def interpolate(T0: np.ndarray, T1: np.ndarray, n: int) -> list[np.ndarray]:
    """n poses strictly after T0 up to and including T1 (linear position, slerp rotation)."""
    if n <= 0:
        return []
    slerp = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))
    out = []
    for w in np.linspace(0, 1, n + 1)[1:]:
        out.append(pose_matrix((1 - w) * T0[:3, 3] + w * T1[:3, 3], slerp(w).as_matrix()))
    return out


def transformed_segment(
    src: SourceDemo, phase: Phase, env: SceneEnv, transforms: dict | None = None
) -> tuple[dict, np.ndarray]:
    """transforms: side -> (ref, 4x4) from the previous phase. An arm that keeps the same reference keeps
    its transform, so motion stays continuous (re-anchoring on an object that both hands are holding
    makes the arms twitch and can tear the grasp)."""
    a, b = src.boundaries[phase.name]
    transforms = {} if transforms is None else transforms
    seg = {}
    for s in SIDES:
        poses = src.target_pose[s][a:b]
        ref = phase.refs.get(s)
        if ref is None:
            seg[s] = poses.copy()
            transforms.pop(s, None)
            continue
        if transforms.get(s, (None,))[0] != ref:
            # maps the source object pose at the phase start onto the object's current pose
            transforms[s] = (ref, env.object_pose(ref) @ np.linalg.inv(src.object_pose[ref][a]))
        seg[s] = np.einsum("ij,tjk->tik", transforms[s][1], poses)
    return seg, src.openings[a:b]


def generate_episode(
    env: SceneEnv, src: SourceDemo, phases: list[Phase], object_poses: dict, cfg: GenConfig, rng
) -> tuple[Episode, bool, dict]:
    env.set_state(src.initial_state)  # same arm configuration as the source demo
    for name, T in object_poses.items():
        env.set_object_pose(T, name)
    for _ in range(5):
        env._physics_step()
    rec = Recorder(env)
    current = {s: env.eef_pose(s) for s in SIDES}
    openings = {s: float(src.openings[0, i]) for i, s in enumerate(SIDES)}
    info = {}

    def run(targets_seq: list[dict], openings_seq: list[dict]) -> None:
        for targets, ops in zip(targets_seq, openings_seq):
            cmd = {}
            for s in SIDES:
                T = targets[s].copy()
                if cfg.action_noise_m:
                    T[:3, 3] += rng.normal(0, cfg.action_noise_m, 3)
                cmd[s] = T
            rec.before_step()
            arm = env.step_ee(cmd, ops)
            rec.after_command(cmd, arm, ops)

    boundaries, transforms = {}, {}
    for phase in phases:
        seg, seg_open = transformed_segment(src, phase, env, transforms)
        if len(seg_open) == 0:
            continue
        start = len(rec.ep)
        interp = {s: interpolate(current[s], seg[s][0], cfg.interp_steps) for s in SIDES}
        run(
            [{s: interp[s][i] for s in SIDES} for i in range(cfg.interp_steps)],
            [dict(openings) for _ in range(cfg.interp_steps)],
        )
        run(
            [{s: seg[s][t] for s in SIDES} for t in range(len(seg_open))],
            [{s: float(seg_open[t, i]) for i, s in enumerate(SIDES)} for t in range(len(seg_open))],
        )
        current = {s: seg[s][-1] for s in SIDES}
        openings = {s: float(seg_open[-1, i]) for i, s in enumerate(SIDES)}
        info[f"{phase.name}_held"] = {n: sorted(env.holders(n)) for n in env.object_names}
        boundaries[phase.name] = [start, len(rec.ep)]

    run([dict(current) for _ in range(cfg.hold_steps)], [dict(openings) for _ in range(cfg.hold_steps)])
    if phases[-1].name in boundaries:
        boundaries[phases[-1].name][1] = len(rec.ep)  # the final hold belongs to the last phase
    info["boundaries"] = boundaries
    return rec.ep, env.is_success(), info
