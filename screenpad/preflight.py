"""
`screenpad preflight`: check a machine before scoring with it.
--------------------------------------------------------------

Each check prints PASS, WARN or FAIL; the command exits non-zero if any check fails, so it can gate a run (`screenpad preflight && screenpad infer ...`).

1. Environment: Python and the key libraries import, with their versions.
2. Ensemble: the frozen ensemble file loads; its SHA-256, members and threshold.
3. Weights: every weight file the ensemble needs is present (downloaded once from the public HF repo if not, unless --no-download).
4. Face detector: YuNet loads and runs.
5. Golden test: two fixed synthetic images (seeded, generated in code, so no dataset image ships) go through the full inference path (load, detect, crop, score); each member's z and the ensemble z must match the reference recorded for this ensemble (artefacts/preflight_reference.json) within --tolerance. This catches wrong or corrupt weights, a broken install or numerical drift on new hardware.
6. Input scan (with --input): files by format, unsupported or unreadable files, images below 128 px, 16-bit images: problems are visible before scoring.

`--write-reference` records the golden scores for the current ensemble on this machine (done once, by the author).
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from .cli_common import _make_cfg, _setup

REFERENCE = Path("artefacts/preflight_reference.json")


def golden_images() -> dict[str, np.ndarray]:
    """Two fixed 512 px RGB inputs: seeded noise (attack-like texture) and a smooth gradient with mild noise (camera-like)."""
    rng = np.random.default_rng(20261021)
    noise = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:512, 0:512] / 511.0
    smooth = np.stack([0.25 + 0.5 * xx, 0.3 + 0.4 * yy, 0.35 + 0.3 * (1 - xx)], -1) * 255 + rng.normal(0, 2, (512, 512, 3))
    return {"seeded_noise": noise, "smooth_gradient": np.clip(np.rint(smooth), 0, 255).astype(np.uint8)}


class _Report:
    def __init__(self, log):
        self.log, self.rows = log, []

    def add(self, check: str, status: str, detail: str):
        self.rows.append((check, status, detail))
        (self.log.error if status == "FAIL" else self.log.warning if status == "WARN" else self.log.info)("%-14s %-4s %s", check, status, detail)

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for _, s, _ in self.rows)


def _scan(folder: Path, recursive: bool) -> dict:
    from PIL import Image

    from .data.io import list_images
    from .infer import unsupported_files

    images = list_images(folder, recursive=recursive)
    out = {"images": len(images), "by_format": {}, "unsupported": [str(p.relative_to(folder)) for p in unsupported_files(folder, recursive, images)],
           "unreadable": [], "small": [], "high_bit_depth": []}
    for p in images:
        ext = p.suffix.lower()
        out["by_format"][ext] = out["by_format"].get(ext, 0) + 1
        try:
            with Image.open(p) as im:
                im.verify()  # header and structure, without decoding all pixels
            with Image.open(p) as im:
                if min(im.size) < 128:
                    out["small"].append(f"{p.relative_to(folder)} ({im.size[0]}x{im.size[1]})")
                if im.mode.startswith("I") or im.mode == "F":
                    out["high_bit_depth"].append(str(p.relative_to(folder)))
                im.load()  # full decode catches truncated files
        except Exception as e:  # noqa: BLE001 - any decode failure is reported, not raised
            out["unreadable"].append(f"{p.relative_to(folder)} ({type(e).__name__})")
    return out


def cmd_preflight(args) -> None:
    import importlib

    cfg = _make_cfg(args)
    log = _setup(cfg)
    r = _Report(log)

    # 1. environment
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "torch", "timm", "cv2", "PIL", "pandas", "huggingface_hub"):
        try:
            versions[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception as e:  # noqa: BLE001
            r.add("environment", "FAIL", f"cannot import {mod}: {e}")
    r.add("environment", "PASS", ", ".join(f"{k} {v}" for k, v in versions.items()) + f"; {platform.machine()}")

    # 2. ensemble file
    frozen_path = Path(args.frozen)
    try:
        sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        frozen = json.loads(frozen_path.read_text())
        r.add("ensemble", "PASS", f"{frozen_path} sha256 {sha[:12]}; members {frozen['enabled']}; threshold z {frozen['threshold_z']:.3f}; calibrated on {frozen.get('calibrated_on', {}).get('device', '?')}")
    except Exception as e:  # noqa: BLE001
        r.add("ensemble", "FAIL", f"cannot read {frozen_path}: {e}")
        raise SystemExit(1)

    # 3. weights
    from .infer import _has_weights, resolve_frozen

    if _has_weights(frozen):
        ready = frozen
    elif args.no_download and not (Path(args.models) / "frozen.json").exists():
        r.add("weights", "FAIL", f"not found locally and --no-download given (run `screenpad fetch-models --dest {args.models}`)")
        raise SystemExit(1)
    else:
        ready = resolve_frozen(frozen_path, Path(args.models), log)
    if _has_weights(ready):
        r.add("weights", "PASS", f"all files for {ready['enabled']} present under {Path(ready['roots']['runs']).resolve()}")
    else:
        r.add("weights", "FAIL", "weight files missing after download")
        raise SystemExit(1)

    # 4. face detector
    from .data.preprocess import FaceCropper

    try:
        cropper = FaceCropper(det_size=int(cfg.preprocess.det_size), align=bool(cfg.preprocess.align))
        cropper.detect(np.zeros((480, 640, 3), np.uint8))
        r.add("face detector", "PASS", "YuNet loads and runs")
    except Exception as e:  # noqa: BLE001
        r.add("face detector", "FAIL", f"{type(e).__name__}: {e}")
        raise SystemExit(1)

    # 5. golden test, through the same path as `infer`
    from .ensemble.scorer import EnsembleScorer
    from .utils.device import resolve_device

    device = resolve_device(cfg.device)
    ready["members"] = {n: ready["members"][n] for n in ready["enabled"]}
    scorer = EnsembleScorer(ready, device, args.batch_size)
    p = cfg.preprocess
    variants = {c: float(p.crops[c]) for c in sorted(scorer.crops)}
    gold = golden_images()
    crops = {c: np.stack([cropper.crop(img, cropper.detect(img), k, int(p.size))[0] for img in gold.values()]) for c, k in variants.items()}
    s = scorer.score(crops)
    got = {name: {m: float(s[m].iloc[i]) for m in [*ready["enabled"], "ensemble_z"]} for i, name in enumerate(gold)}
    ref_path = Path(args.reference)
    if args.write_reference:
        ref_path.write_text(json.dumps({"frozen_sha256": sha, "device": str(device), "machine": platform.machine(), "scores": got}, indent=2))
        r.add("golden test", "PASS", f"reference written to {ref_path} ({device}, {platform.machine()})")
    elif not ref_path.exists():
        r.add("golden test", "WARN", f"no reference at {ref_path}; scores {got}")
    else:
        ref = json.loads(ref_path.read_text())
        if ref.get("frozen_sha256") != sha:
            r.add("golden test", "WARN", f"reference was recorded for another ensemble (sha256 {ref.get('frozen_sha256', '?')[:12]}); not compared")
        else:
            diff = max(abs(got[n][m] - ref["scores"][n][m]) for n in got for m in got[n])
            status = "PASS" if diff <= args.tolerance else "FAIL"
            r.add("golden test", status, f"max |z - reference| = {diff:.4f} (tolerance {args.tolerance}; reference from {ref.get('device')}, {ref.get('machine')}; here {device}, {platform.machine()})")

    # 6. input scan
    if args.input:
        folder = Path(args.input)
        if not folder.is_dir():
            r.add("input", "FAIL", f"{folder} is not a folder")
        else:
            sc = _scan(folder, args.recursive)
            r.add("input", "PASS" if sc["images"] else "FAIL", f"{sc['images']} images {sc['by_format']}")
            for key, label in (("unsupported", "not a supported format (row with an error, no score)"), ("unreadable", "cannot be decoded (row with an error, no score)"),
                               ("small", "smaller than 128 px (upsampled: scores tend to be attack-like)"), ("high_bit_depth", "16-bit or float (rescaled to 8-bit)")):
                if sc[key]:
                    r.add("input", "WARN", f"{len(sc[key])} {label}: {', '.join(sc[key][:8])}{' ...' if len(sc[key]) > 8 else ''}")

    log.info("preflight: %s", "FAILED" if r.failed else "OK")
    if r.failed:
        raise SystemExit(1)
