"""
Report-level PAD rates with exact confidence intervals, per phone, and DET curve points.
----------------------------------------------------------------------------------------

Scores are P(attack) or any monotone attack score: higher = more likely an attack.

Why exact intervals: with 17 attacks, catching all 17 still only bounds the true APCER below about 19.5% (Clopper-Pearson 95%
upper bound for 0/17), so a headline "0% APCER" on a small attack set says very little. Every rate here carries its
Clopper-Pearson interval, computed from the exact binomial rather than a normal approximation.

APCER: fraction of attacks accepted as bona fide (score < threshold).
BPCER: fraction of bona fide rejected as attacks (score >= threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_curve

from ..constants import ATTACK, BONAFIDE

PHONE_NAMES = {"iPhone 15": "iPhone 15", "SM-A127F": "Galaxy A12"}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) binomial CI for k/n. Returns (nan, nan) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def rates_at_threshold(y, score, threshold: float) -> dict:
    """APCER and BPCER at a threshold, with Clopper-Pearson 95% intervals and the raw miss/reject counts."""
    y, score = np.asarray(y), np.asarray(score)
    attack, bona = score[y == ATTACK], score[y == BONAFIDE]
    n_att, n_bf = len(attack), len(bona)
    k_att = int((attack < threshold).sum())
    k_bf = int((bona >= threshold).sum())
    apcer = k_att / n_att if n_att else float("nan")
    bpcer = k_bf / n_bf if n_bf else float("nan")
    a_lo, a_hi = clopper_pearson(k_att, n_att)
    b_lo, b_hi = clopper_pearson(k_bf, n_bf)
    return {
        "n_attack": n_att,
        "n_bonafide": n_bf,
        "apcer": apcer,
        "apcer_lo95": a_lo,
        "apcer_hi95": a_hi,
        "bpcer": bpcer,
        "bpcer_lo95": b_lo,
        "bpcer_hi95": b_hi,
        "attacks_missed": k_att,
        "bonafide_rejected": k_bf,
    }


def by_group(df: pd.DataFrame, threshold: float, group_col: str, bonafide_shared: bool = True) -> pd.DataFrame:
    """
    One row per value of `group_col` (column `group`), plus a final row with group "all".

    df needs columns `label` (0 bona fide / 1 attack), `score` and `group_col`.

    bonafide_shared=True (default): bona fide have no phone, so every group's BPCER is computed on ALL bona fide in df,
    and only the attack side is split by group. Groups are taken from attack rows only, so bona fide rows do not add
    a meaningless group. The BPCER columns are then identical across groups and equal to the "all" row.

    bonafide_shared=False: each group uses only its own rows. A group with no bona fide gets nan bona fide fields.
    """
    bona_all = df[df.label == BONAFIDE]
    if bonafide_shared:
        groups = df.loc[df.label == ATTACK, group_col].dropna().unique()
    else:
        groups = df[group_col].dropna().unique()
    rows = []
    for g in groups:
        sub = df[df[group_col] == g]
        if bonafide_shared:
            sub = pd.concat([sub[sub.label == ATTACK], bona_all])
        r = rates_at_threshold(sub.label.to_numpy(), sub.score.to_numpy(), threshold)
        rows.append({"group": g, **r})
    rows.append({"group": "all", **rates_at_threshold(df.label.to_numpy(), df.score.to_numpy(), threshold)})
    return pd.DataFrame(rows)


def phone_of(ids, manifest_csv) -> np.ndarray:
    """Friendly phone name of the attack recapture for each id ("" when there is no attack row or no EXIF model)."""
    m = pd.read_csv(manifest_csv, dtype={"id": str})
    attacks = m[m.label == ATTACK].drop_duplicates("id").set_index("id")["exif_model"]
    out = []
    for i in ids:
        model = attacks.get(str(i), "")
        model = "" if pd.isna(model) else str(model)
        out.append(PHONE_NAMES.get(model, model))
    return np.array(out, dtype=object)


def det_points(y, score) -> pd.DataFrame:
    """
    APCER/BPCER at every distinct score threshold, in sklearn roc_curve order: threshold descending.

    Down the rows, apcer is non-increasing and bpcer is non-decreasing. The first row is the +inf threshold (apcer 1, bpcer 0).
    """
    fpr, tpr, thr = roc_curve(np.asarray(y), np.asarray(score))
    return pd.DataFrame({"threshold": thr, "apcer": 1 - tpr, "bpcer": fpr})
