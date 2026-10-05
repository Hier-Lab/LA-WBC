"""Download the LA-WBC simulation dataset from the Hugging Face Hub.

Example::

    python scripts/download_dataset.py                      # both subsets -> ./dataset
    python scripts/download_dataset.py --subset w_cont_id   # one subset only

The resulting layout is::

    dataset/<subset>/<run_id>/
        index.csv, manifest.json
        episodes/epXXXXXX/{data.h5, meta.json}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "2Nitrogen/LA-WBC"
SUBSETS = ("w_cont_id", "wo_cont_id")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID,
                   help="Hugging Face dataset repository")
    p.add_argument("--subset", type=str, choices=SUBSETS + ("all",), default="all")
    p.add_argument("--local_dir", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "dataset"))
    p.add_argument("--max_workers", type=int, default=8)
    args = p.parse_args()

    subsets = SUBSETS if args.subset == "all" else (args.subset,)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=[f"{s}/**" for s in subsets],
        max_workers=args.max_workers,
    )
    print(f"Downloaded {', '.join(subsets)} to {path}")


if __name__ == "__main__":
    main()
