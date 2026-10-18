"""
PAD metrics (ISO/IEC 30107-3 vocabulary). Scores are P(attack): higher = more likely an attack.

APCER: fraction of attacks accepted as bona fide (score < threshold).
BPCER: fraction of bona fide rejected as attacks (score >= threshold).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def apcer_bpcer(y: np.ndarray, score: np.ndarray, threshold: float) -> tuple[float, float]:
    y, score = np.asarray(y), np.asarray(score)
    apcer = float((score[y == 1] < threshold).mean()) if (y == 1).any() else float("nan")
    bpcer = float((score[y == 0] >= threshold).mean()) if (y == 0).any() else float("nan")
    return apcer, bpcer


def eer(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def bpcer_at_apcer(y: np.ndarray, score: np.ndarray, apcer_target: float) -> float:
    """Lowest BPCER achievable while keeping APCER <= apcer_target."""
    fpr, tpr, _ = roc_curve(y, score)
    ok = (1 - tpr) <= apcer_target
    return float(fpr[ok].min()) if ok.any() else 1.0


def summary(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    apcer, bpcer = apcer_bpcer(y, score, threshold)
    return {
        "n_bonafide": int((np.asarray(y) == 0).sum()),
        "n_attack": int((np.asarray(y) == 1).sum()),
        "auc": float(roc_auc_score(y, score)),
        "eer": eer(y, score),
        f"apcer@{threshold:g}": apcer,
        f"bpcer@{threshold:g}": bpcer,
        "acer": (apcer + bpcer) / 2,
        "bpcer@apcer10%": bpcer_at_apcer(y, score, 0.10),
    }
