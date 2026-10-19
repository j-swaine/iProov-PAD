"""
Ensemble: calibrate each member against bona fide faces only, average with equal weights, and measure every coalition of members (ablation).
-------------------------------------------------------------------------------------------------------------------------------------------

Members are independent and switchable (`ensemble.members.<name>.enabled`):
- `run`: a fine-tuned configuration, one or more seeds (their calibrated scores are averaged into one member);
- `oneclass`: PatchCore or Mahalanobis (results/oneclass/<name>/scores.csv).

Calibration (bona fide only, so none of the 6 real dev attacks is spent on it). Each score is mapped to a scale where higher means more attack-like and tails are kept: logit for a classifier probability, log for a one-class distance. It is then robust-z-scored against the calibration bona fide (val originals plus camera-only controls, none seen in training): z = (t - median) / (1.4826 * MAD), i.e. "how many typical bona fide spreads above typical bona fide". A percentile would cap at 100 and tie every clear attack. Combination is by family, not by member: the two-class members (fine-tuned classifiers, which know the simulated attacks) are averaged into one score, the one-class members (which know only bona fide) into another, and the final score is the mean of the families. The one-class side therefore keeps half the decision however many classifiers are added: it is the only defence against attack types the simulator does not produce. `combine` also offers `mean` (every member equal) and `max` (flag if either family flags) for comparison. Nothing is fitted to attacks.

Threshold: the ensemble z at which a `bpcer_target` fraction of calibration bona fide is rejected. Fixed on bona fide only, reported on the other sets.

Ablation (`ablation`): every non-empty coalition of the candidate members, scored on dev with one shared stratified bootstrap (attacks and bona fide resampled separately), so coalitions are compared on the same resamples: AUC with 95% CI, BPCER@APCER0, APCER/BPCER at the bona fide threshold, and the paired AUC difference to the best coalition. `shapley` gives each member's average marginal contribution to dev AUC over all coalitions.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["set", "id", "kind", "variant", "label"]


def _transform(s: np.ndarray, kind: str) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    if kind == "run":
        s = np.clip(s, 1e-7, 1 - 1e-7)
        return np.log(s / (1 - s))
    return np.log(np.maximum(s, 1e-12))


def robust_z(t: np.ndarray, ref: np.ndarray) -> np.ndarray:
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med))
    return (t - med) / max(mad, 1e-9)


def calibration_mask(df: pd.DataFrame) -> np.ndarray:
    """Val bona fide: originals and camera-only controls (never trained on, never attacks)."""
    return ((df.set == "val") & (df.label == 0)).to_numpy()


def load_member(spec: dict, roots: dict) -> pd.DataFrame:
    """KEY columns + one `z` column per source file (seed), calibrated; `z` is their mean."""
    if spec["type"] == "run":
        paths = [Path(roots["runs"]) / r / "scores_last.csv" for r in spec["runs"]]
    else:
        paths = [Path(roots["oneclass"]) / spec["path"] / "scores.csv"]
    out = None
    for i, p in enumerate(paths):
        df = pd.read_csv(p, dtype={"id": str})
        df = df[df.set.isin(["val", "dev"])][KEY + ["score"]].copy()
        t = _transform(df.score.to_numpy(), spec["type"])
        df[f"z{i}"] = robust_z(t, t[calibration_mask(df)])
        df = df.drop(columns="score")
        out = df if out is None else out.merge(df, on=KEY, how="inner", validate="one_to_one")
    zc = [c for c in out.columns if c.startswith("z")]
    out["z"] = out[zc].mean(1)
    return out[KEY + ["z"]]


def member_table(members: dict, roots: dict) -> pd.DataFrame:
    """One row per image (val + dev), one column per member: calibrated z."""
    tab = None
    for name, spec in members.items():
        m = load_member(spec, roots).rename(columns={"z": name})
        tab = m if tab is None else tab.merge(m, on=KEY, how="inner", validate="one_to_one")
    return tab


FAMILY = {"run": "twoclass", "oneclass": "oneclass"}


def combine(df: pd.DataFrame, members: list[str], families: dict[str, str], mode: str = "family") -> np.ndarray:
    """Ensemble z for `members`: `family` = mean of per-family means; `mean` = mean of members; `max` = max of per-family means."""
    if mode == "mean":
        return df[members].mean(axis=1).to_numpy()
    fam = {}
    for m in members:
        fam.setdefault(families[m], []).append(m)
    per = np.stack([df[ms].mean(axis=1).to_numpy() for ms in fam.values()], 1)
    return per.max(1) if mode == "max" else per.mean(1)


def threshold(z_cal: np.ndarray, bpcer_target: float) -> float:
    return float(np.quantile(z_cal, 1 - bpcer_target))


def _auc(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUC per row for (B, n_pos) and (B, n_neg) score arrays (ties count half)."""
    gt = (pos[:, :, None] > neg[:, None, :]).mean((1, 2))
    eq = (pos[:, :, None] == neg[:, None, :]).mean((1, 2))
    return gt + 0.5 * eq


def ablation(tab: pd.DataFrame, candidates: list[str], families: dict[str, str], bpcer_target: float, seed: int,
             mode: str = "family", n_boot: int = 2000) -> pd.DataFrame:
    dev = tab[tab.set == "dev"]
    cal = tab[calibration_mask(tab)]
    y = dev.label.to_numpy()
    pos_i, neg_i = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    bp = rng.choice(pos_i, (n_boot, len(pos_i)))
    bn = rng.choice(neg_i, (n_boot, len(neg_i)))
    rows, boots = [], {}
    for k in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, k):
            s = combine(dev, list(combo), families, mode)
            thr = threshold(combine(cal, list(combo), families, mode), bpcer_target)
            a = s[pos_i]
            b = s[neg_i]
            boots[combo] = _auc(s[bp], s[bn])
            rows.append({"coalition": "+".join(combo), "size": k, "combine": mode, "dev_auc": float(_auc(a[None], b[None])[0]),
                         "dev_auc_lo95": float(np.percentile(boots[combo], 2.5)), "dev_auc_hi95": float(np.percentile(boots[combo], 97.5)),
                         "dev_bpcer@apcer0": float((b >= a.min()).mean()), "threshold_z": thr,
                         "dev_apcer@thr": float((a < thr).mean()), "dev_bpcer@thr": float((b >= thr).mean())})
    out = pd.DataFrame(rows)
    best = out.sort_values(["dev_auc", "dev_bpcer@apcer0"], ascending=[False, True]).iloc[0].coalition
    bb = boots[tuple(best.split("+"))]
    diff = {c: boots[c] - bb for c in boots}
    out["delta_auc_vs_best_lo95"] = [float(np.percentile(diff[tuple(c.split("+"))], 2.5)) for c in out.coalition]
    out["delta_auc_vs_best_hi95"] = [float(np.percentile(diff[tuple(c.split("+"))], 97.5)) for c in out.coalition]
    out["within_ci_of_best"] = out.delta_auc_vs_best_hi95 >= 0
    return out


def shapley(abl: pd.DataFrame, candidates: list[str], metric: str = "dev_auc") -> pd.DataFrame:
    """Each member's average marginal contribution to `metric` over all coalitions (empty coalition = 0.5 AUC)."""
    from math import factorial

    v = {frozenset(c.split("+")): m for c, m in zip(abl.coalition, abl[metric])}
    v[frozenset()] = 0.5
    n = len(candidates)
    rows = []
    for m in candidates:
        others = [c for c in candidates if c != m]
        phi = 0.0
        for k in range(n):
            for s in itertools.combinations(others, k):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                phi += w * (v[frozenset(s) | {m}] - v[frozenset(s)])
        full = frozenset(candidates)
        rows.append({"member": m, f"shapley_{metric}": phi, f"leave_one_out_{metric}": v[full] - v[full - {m}] if len(candidates) > 1 else np.nan,
                     f"alone_{metric}": v[frozenset({m})]})
    return pd.DataFrame(rows)
