"""
Resampling counterfactual ("shrink test"): does a model's score depend on how hard a crop was downsampled?
-------------------------------------------------------------------------------------------------------

Real attack faces are ~3x more pixels wide than bona fide ones, so their crops are downsampled ~6x versus ~2.4x, and in training only original bona fide crops sit below ~3x. A model could learn "lightly resampled -> bona fide" instead of the screen.

The test holds the content fixed and changes only the resampling history: each bona fide val/dev photo is resized by a factor f (area when shrinking, cubic when enlarging) before the normal crop, reusing the face detection from the unresized photo (scaled by f) so framing is identical. The crop's resample ratio becomes ~2.4 * f; f = 2.5 matches the real attacks. f = 1 reproduces the cached crop exactly.

Read-out per factor (`shrink_metrics`): the median percentile of the resized crops' scores within the f = 1 scores of the same faces. A model that ignores resampling stays near 50; one that uses it climbs towards 100 as f grows.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.io import load_image
from ..data.preprocess import Detection, FaceCropper

FACTORS = (0.5, 0.75, 1.0, 1.5, 2.5, 4.0)
KIND = "shrink_cf"
SPLIT = "shrink"


def _scaled(det: Detection | None, f: float) -> Detection | None:
    return None if det is None else Detection(box=det.box * f, landmarks=det.landmarks * f, score=det.score, n_faces=det.n_faces)


def build_shrink_set(splits: pd.DataFrame, root: str | Path, k: float, size: int, det_size: int, align: bool,
                     factors=FACTORS) -> dict:
    """Crop cache of every val/dev bona fide photo at each resize factor. Its split is `shrink`, so no other set picks it up."""
    rows = splits[(splits.label == 0) & splits.split.isin(["val", "dev"])].reset_index(drop=True)
    cropper = FaceCropper(det_size=det_size, align=align)
    images, info, meta = [], [], []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="shrink", mininterval=10):
        rgb = np.asarray(load_image(Path(root) / r.path))
        det = cropper.detect(rgb)
        for v, f in enumerate(factors):
            src = rgb if f == 1 else cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
            img, ci = cropper.crop(src, _scaled(det, f), k, size)
            images.append(img)
            info.append(ci.to_dict())
            meta.append((r.id, r.path, v, f))
    info = pd.DataFrame(info)
    ids, paths, variants, fs = zip(*meta)
    n = len(images)
    return {"images": np.stack(images), "id": np.asarray(ids, dtype=str), "path": np.asarray(paths, dtype=str),
            "split": np.full(n, SPLIT), "label": np.zeros(n, np.int8), "kind": np.full(n, KIND),
            "variant": np.asarray(variants), "factor": np.asarray(fs, dtype=np.float32),
            **{f"info_{c}": info[c].to_numpy() for c in info.columns}, "k": np.float32(k), "size": np.int32(size), "align": np.bool_(align)}


def shrink_name(crop: str, size: int) -> str:
    return f"shrink_{crop}_{size}_aligned.npz"


def shrink_metrics(scores: pd.DataFrame, factors=FACTORS) -> dict:
    """`scores`: one row per shrink crop with id, variant and score. Returns shrink_pct@<f> for each factor != 1."""
    piv = scores.pivot_table(index="id", columns="variant", values="score")
    base = np.sort(piv[factors.index(1.0)].to_numpy())
    out = {}
    for v, f in enumerate(factors):
        if f != 1.0:
            pct = np.searchsorted(base, piv[v].to_numpy(), side="right") / len(base) * 100
            out[f"shrink_pct@{f:g}"] = float(np.median(pct))
    return out


def detectability(feats: np.ndarray, ids: np.ndarray, factors: np.ndarray, levels=FACTORS) -> dict:
    """Positive control: grouped-CV AUC of a linear probe telling each factor's crops from the factor-1 crops of the same faces.

    Near 0.5 means the resized crops are indistinguishable from the originals in these features, so a flat shrink test is uninformative at that factor; well above 0.5 means the information is there and a flat shrink test shows the model does not use it.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    for f in levels:
        if f == 1.0:
            continue
        m = (factors == f) | (factors == 1.0)
        y = (factors[m] == f).astype(int)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
        p = cross_val_predict(clf, feats[m], y, groups=ids[m], cv=GroupKFold(5), method="predict_proba")[:, 1]
        out[f"detectable_auc@{f:g}"] = float(roc_auc_score(y, p))
    return out
