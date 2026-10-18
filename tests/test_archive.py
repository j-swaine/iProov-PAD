"""The raw-data tar must round-trip byte-for-byte, deterministically, and catch tampering."""

from __future__ import annotations

import pytest
from PIL import Image

from screenpad.data.archive import build_archive, extract_archive, read_checksums, sha256_file, write_checksums


@pytest.fixture
def raw_root(tmp_path):
    root = tmp_path / "raw"
    for folder, colour in (("Bonafide", "red"), ("Screens", "blue")):
        (root / folder).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (8 + i, 8), colour).save(root / folder / f"{i:05d}.png")
    return root


def test_round_trip_is_byte_identical(raw_root, tmp_path):
    sums = build_archive(raw_root, tmp_path / "a.tar")
    write_checksums(sums, tmp_path / "a.sha256")
    assert read_checksums(tmp_path / "a.sha256") == sums
    assert extract_archive(tmp_path / "a.tar", tmp_path / "out", sums) == 6
    for rel in sums:
        assert (tmp_path / "out" / rel).read_bytes() == (raw_root / rel).read_bytes()


def test_build_is_deterministic(raw_root, tmp_path):
    build_archive(raw_root, tmp_path / "a.tar")
    (raw_root / "Bonafide" / "00000.png").touch()  # new mtime must not change the archive
    build_archive(raw_root, tmp_path / "b.tar")
    assert sha256_file(tmp_path / "a.tar") == sha256_file(tmp_path / "b.tar")


def test_tampered_checksum_is_caught(raw_root, tmp_path):
    sums = build_archive(raw_root, tmp_path / "a.tar")
    sums["Screens/00000.png"] = "0" * 64
    with pytest.raises(ValueError, match="1 hash mismatches"):
        extract_archive(tmp_path / "a.tar", tmp_path / "out", sums)
