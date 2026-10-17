"""Download example MimicGen datasets from the Hugging Face Hub.

The official MimicGen release (https://huggingface.co/datasets/amandlek/mimicgen_datasets) contains
1000 generated demos per task in robomimic HDF5 format. Files are 1-13 GB, so use --list to see sizes.

    uv run at-download --list
    uv run at-download core/stack_d1.hdf5
"""

import argparse

from huggingface_hub import HfApi, hf_hub_download

import adapter_training  # noqa: F401  (TLS setup)
from adapter_training.common import RAW_DIR

MIMICGEN_REPO = "amandlek/mimicgen_datasets"


def list_files(repo_id: str) -> None:
    files = HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    rows = sorted((f.path, f.size) for f in files if f.path.endswith(".hdf5"))
    for path, size in rows:
        print(f"{size / 1e9:7.2f} GB  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", default="core/stack_d1.hdf5", help="Path inside the dataset repo")
    parser.add_argument("--repo-id", default=MIMICGEN_REPO)
    parser.add_argument("--list", action="store_true", help="List available HDF5 files and sizes")
    args = parser.parse_args()

    if args.list:
        list_files(args.repo_id)
        return

    path = hf_hub_download(args.repo_id, args.file, repo_type="dataset", local_dir=RAW_DIR)
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
