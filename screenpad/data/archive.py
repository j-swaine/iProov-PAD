"""
Single-file raw data archive.
-----------------------------

The raw dataset goes to the HF Hub as one uncompressed tar of the original files plus a SHA-256 list, instead of ~1000 loose files (which the Hub serves at a few files per second). The tar holds the files byte-for-byte, so EXIF/ICC metadata survives for the shortcut audit, and `extract_archive` re-hashes every file so a round trip is provably exact.

The tar is built deterministically (sorted members, zeroed owner/mtime), so rebuilding it from the same files gives the same archive hash.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

from ..constants import CLASS_FOLDERS
from .io import list_images


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _members(root: Path) -> list[Path]:
    """Every image under root/<class folder>/, as paths relative to root, sorted."""
    return sorted(p.relative_to(root) for folder in CLASS_FOLDERS for p in list_images(root / folder))


def build_archive(root: str | Path, tar_path: str | Path) -> dict[str, str]:
    """Write `tar_path` from the class folders under `root` and return {relative path: sha256}."""
    root = Path(root)
    checksums = {}

    def _normalise(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode = 0o644
        return info

    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tar:
        for rel in _members(root):
            tar.add(root / rel, arcname=rel.as_posix(), filter=_normalise)
            checksums[rel.as_posix()] = sha256_file(root / rel)
    return checksums


def write_checksums(checksums: dict[str, str], path: str | Path) -> None:
    """`sha256sum`-compatible listing, so `shasum -a 256 -c` can also verify it."""
    Path(path).write_text("".join(f"{h}  {p}\n" for p, h in sorted(checksums.items())))


def read_checksums(path: str | Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        h, p = line.split("  ", 1)
        out[p] = h
    return out


def extract_archive(tar_path: str | Path, dest: str | Path, checksums: dict[str, str]) -> int:
    """Extract `tar_path` into `dest` and verify every file against `checksums`.

    Raises ValueError if any file is missing, unexpected, or differs. Returns the number of files verified.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
        tar.extractall(dest, filter="data")
    missing, extra = set(checksums) - names, names - set(checksums)
    bad = [p for p in sorted(checksums.keys() & names) if sha256_file(dest / p) != checksums[p]]
    if missing or extra or bad:
        raise ValueError(
            f"archive check failed: {len(missing)} missing, {len(extra)} unexpected, {len(bad)} hash mismatches"
            f" (e.g. {sorted(missing | extra)[:3] + bad[:3]})"
        )
    return len(checksums)
