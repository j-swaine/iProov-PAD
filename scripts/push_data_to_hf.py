"""Upload the supplied dataset to a private Hugging Face dataset repo as one tar + SHA-256 list, so `screenpad pull-data` can fetch it in a single download.

Usage:
    uv run python scripts/push_data_to_hf.py [--root DATA/test/data] [--repo Jamie1701/screenpad-data]

The repo is created private. The bona fide images appear to be FFHQ (CC BY-NC-SA 4.0) and the screen captures are the assessor's, so the repo must stay private.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from screenpad.data.archive import build_archive, extract_archive, sha256_file, write_checksums

CARD = """---
license: other
viewer: false
---
# screenpad-data (private)

Take-home PAD dataset: 1000 bona fide face images (`Bonafide/`, apparently FFHQ, CC BY-NC-SA 4.0) and 23 iPhone recaptures of the bona fide images with the same filename (`Screens/`). Private; do not redistribute.

- `raw/screenpad-raw.tar` - the original files, byte-identical, in one uncompressed tar. Fetch with `screenpad pull-data`.
- `raw/screenpad-raw.sha256` - per-file SHA-256 (`shasum -a 256 -c` format); `pull-data` verifies every file against it.
- `data/` - the same files uploaded loose (initial upload; slow to fetch, kept for browsing).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="DATA/test/data")
    ap.add_argument("--repo", default="Jamie1701/screenpad-data")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tar_path, sums_path = tmp / "screenpad-raw.tar", tmp / "screenpad-raw.sha256"
        checksums = build_archive(args.root, tar_path)
        write_checksums(checksums, sums_path)
        # Prove the round trip before uploading anything.
        n = extract_archive(tar_path, tmp / "check", checksums)
        print(f"built {tar_path.name}: {n} files, round trip verified, tar sha256 {sha256_file(tar_path)}")

        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        for f in (tar_path, sums_path):
            api.upload_file(repo_id=args.repo, repo_type="dataset", path_or_fileobj=f, path_in_repo=f"raw/{f.name}")
        api.upload_file(repo_id=args.repo, repo_type="dataset", path_or_fileobj=CARD.encode(), path_in_repo="README.md")
    print(f"uploaded -> https://huggingface.co/datasets/{args.repo}/tree/main/raw")


if __name__ == "__main__":
    main()
