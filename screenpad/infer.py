"""
`screenpad infer`: score a folder of raw face images with the frozen ensemble and write one CSV row per image.
--------------------------------------------------------------------------------------------------------------

The default ensemble is the deployed photometric variant (artefacts/frozen_photo.json); artefacts/frozen.json is the pre-registered system evaluated once on test.

Each image goes through exactly the pipeline the models were trained and evaluated with: `load_image` (EXIF orientation, sRGB, metadata dropped), YuNet face detection (largest face), eye-level aligned square crop at the configured multiple of the face box, one resample to the cache size. The crops are then scored by `EnsembleScorer`, so the score is the calibrated ensemble z and the decision uses the frozen threshold.

Weights: if the frozen ensemble's own paths have no weights (e.g. a fresh clone), they are downloaded once from the public HF runs repo into `--models` (see `ensemble.cli.fetch_models`) and reused afterwards.

Every file in --input gets a row: unreadable files and unsupported formats get an `error` and no score. Output columns: `image` (path relative to --input), `attack_score` (ensemble z: bona fide spreads above typical bona fide; higher = more likely a screen attack), `attack` (1 if attack_score >= threshold), `threshold`, `face_found`, `n_faces`, one `z_<member>` column per member, and `error` for files that could not be read (no score).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cli_common import _make_cfg, _setup
from .constants import IMAGE_EXTENSIONS


def unsupported_files(folder, recursive: bool, images: list[Path]) -> list[Path]:
    """Files under `folder` that are not supported images (hidden files such as .DS_Store ignored)."""
    folder, seen = Path(folder), set(images)
    found = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in found if p.is_file() and p not in seen and not any(part.startswith(".") for part in p.relative_to(folder).parts))


def _has_weights(frozen: dict) -> bool:
    """Are the enabled members' weight files at the paths `frozen` points to?"""
    roots = frozen["roots"]
    for n in frozen["enabled"]:
        spec = frozen["members"][n]
        if spec["type"] == "run":
            if not all((Path(roots["runs"]) / r / f).exists() for r in spec["runs"] for f in ("config.yaml", "ckpt/last.pt")):
                return False
        elif not (Path(spec.get("root") or roots["oneclass"]) / spec["path"] / "bank.pt").exists():
            return False
    return True


def resolve_frozen(frozen_path: Path, models: Path, log) -> dict:
    """The frozen ensemble with paths that hold its weights: as given, else a previous download in `models`, else a fresh download."""
    from .ensemble.cli import fetch_models

    frozen = json.loads(frozen_path.read_text())
    if _has_weights(frozen):
        return frozen
    sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    local = models / "frozen.json"
    if local.exists():
        f = json.loads(local.read_text())
        if f.get("source_sha256") == sha and _has_weights(f):
            return f
    log.info("weights not found locally: downloading the frozen ensemble (~2 GB, once) into %s", models)
    fetch_models(frozen_path, models, log=log)
    f = json.loads(local.read_text())
    f["source_sha256"] = sha  # ties the download to the frozen file it came from
    local.write_text(json.dumps(f, indent=2))
    return f


def cmd_infer(args) -> None:
    from tqdm import tqdm

    from .data.io import list_images, load_image
    from .data.preprocess import FaceCropper
    from .ensemble.scorer import EnsembleScorer
    from .utils.device import resolve_device

    cfg = _make_cfg(args)
    log = _setup(cfg)
    paths = list_images(args.input, recursive=args.recursive)
    skipped = unsupported_files(args.input, args.recursive, paths)  # every file gets a row, scored or not
    if skipped:
        log.warning("%d file(s) not in a supported image format (%s), listed with an error: %s",
                    len(skipped), ", ".join(IMAGE_EXTENSIONS), ", ".join(str(p.relative_to(args.input)) for p in skipped[:10]))
    if not paths:
        raise SystemExit(f"no images found in {args.input}")
    frozen = resolve_frozen(Path(args.frozen), Path(args.models), log)
    frozen["members"] = {n: frozen["members"][n] for n in frozen["enabled"]}  # only what the decision uses
    device = resolve_device(cfg.device)
    cal = frozen.get("calibrated_on", {}).get("device", "")
    if cal and cal.split(":")[0] != device.type:
        log.warning("calibrated on %s, scoring on %s: scores may differ by a few tenths of a z from the calibration device", cal, device.type)
    scorer = EnsembleScorer(frozen, device, args.batch_size)
    p = cfg.preprocess
    variants = {c: float(p.crops[c]) for c in sorted(scorer.crops)}
    cropper = FaceCropper(det_size=int(p.det_size), align=bool(p.align))
    log.info("%d images, members %s, threshold z %.3f, device %s", len(paths), frozen["enabled"], frozen["threshold_z"], device)

    rows, chunks = [], []
    for start in range(0, len(paths), args.chunk):  # bounded memory: crops for one chunk at a time
        crops, ok = {c: [] for c in variants}, []
        for path in tqdm(paths[start : start + args.chunk], desc="crop", mininterval=5):
            rec = {"image": str(path.relative_to(args.input)), "error": ""}
            try:
                rgb = np.asarray(load_image(path))
                det = cropper.detect(rgb)
                for c, k in variants.items():
                    crops[c].append(cropper.crop(rgb, det, k, int(p.size))[0])
                rec.update(face_found=det is not None, n_faces=det.n_faces if det else 0)
                ok.append(len(rows))
            except Exception as e:  # an unreadable file gets a row with no score rather than stopping the run
                rec.update(face_found=False, n_faces=0, error=f"{type(e).__name__}: {e}")
                log.warning("%s: %s", path, rec["error"])
            rows.append(rec)
        if ok:
            s = scorer.score({c: np.stack(v) for c, v in crops.items()})
            chunks.append(s.set_index(pd.Index(ok)))

    for path in skipped:
        rows.append({"image": str(path.relative_to(args.input)), "face_found": False, "n_faces": 0, "error": f"unsupported file type '{path.suffix}'"})
    df = pd.DataFrame(rows)
    if chunks:
        s = pd.concat(chunks)
        df["attack_score"] = s.ensemble_z
        df["attack"] = s.attack.astype("Int64")
        for n in frozen["enabled"]:
            df[f"z_{n}"] = s[n]
    df["threshold"] = float(frozen["threshold_z"])
    cols = ["image", "attack_score", "attack", "threshold", "face_found", "n_faces"] + [c for c in df if c.startswith("z_")] + ["error"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=cols).to_csv(out, index=False, float_format="%.4f")
    scored = df.attack_score.notna() if "attack_score" in df else pd.Series(False, index=df.index)
    log.info("wrote %s: %d scored, %d flagged as attack, %d without a detected face (scored on a centre crop), %d unreadable or unsupported",
             out, int(scored.sum()), int(df.attack.sum()) if "attack" in df else 0,
             int((scored & ~df.face_found.astype(bool)).sum()), int((df.error != "").sum()))
