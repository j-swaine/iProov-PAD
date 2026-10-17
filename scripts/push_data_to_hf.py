"""Upload the supplied dataset to a private Hugging Face dataset repo, so Colab notebooks can pull it with `screenpad pull-data`.

Usage:
    uv run python scripts/push_data_to_hf.py [--root DATA/test/data] [--repo Jamie1701/screenpad-data]

The repo is created private. The bona fide images appear to be FFHQ (CC BY-NC-SA 4.0) and the screen captures are the assessor's, so the repo must stay private.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
license: other
viewer: false
---
# screenpad-data (private)

Take-home PAD dataset: `data/Bonafide/` (1000 face images, apparently FFHQ, CC BY-NC-SA 4.0) and `data/Screens/` (23 iPhone recaptures of the bona fide images with the same filename). Private; do not redistribute.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="DATA/test/data")
    ap.add_argument("--repo", default="Jamie1701/screenpad-data")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=Path(args.root),
        path_in_repo="data",
        allow_patterns=["Bonafide/*", "Screens/*"],
        commit_message="Upload supplied dataset",
    )
    api.upload_file(repo_id=args.repo, repo_type="dataset", path_or_fileobj=CARD.encode(), path_in_repo="README.md")
    print(f"uploaded {args.root} -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
