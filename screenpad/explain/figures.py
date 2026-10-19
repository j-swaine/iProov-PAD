"""Report figures for `screenpad explain` (matplotlib, written to results/explain/figures/)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATTACK, BONAFIDE = "#c2410c", "#2563eb"
MEMBER_COLOURS = ["#2563eb", "#c2410c", "#0f766e", "#7c3aed"]
FREQ_ORDER = ["original", "blur_0.5", "blur_1", "blur_2", "blur_4", "texture_only"]
FC_ORDER = ["original", "face_only", "context_only"]


def make_figures(out: Path, var: pd.DataFrame, summ: pd.DataFrame, members: list[str], oneclass_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

    for test, order, title in (("frequency", FREQ_ORDER, "Fine texture vs visible content"), ("face_context", FC_ORDER, "Face vs surroundings")):
        s = summ[summ.test == test]
        fig, axes = plt.subplots(1, len(members) + 1, figsize=(3.2 * (len(members) + 1), 3), squeeze=False)
        ax = axes[0, 0]
        for k, m in enumerate(members):
            v = s[s.member == m].set_index("condition").loc[order]
            ax.plot(range(len(order)), v.dev_auc, marker="o", lw=2, color=MEMBER_COLOURS[k], label=m)
        ax.axhline(0.5, color="grey", lw=1, ls=":")
        ax.set(xticks=range(len(order)), ylim=(0.3, 1.02), ylabel="dev AUC", title="Separation (AUC)")
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.legend(frameon=False)
        for k, m in enumerate(members):
            ax = axes[0, k + 1]
            v = s[s.member == m].set_index("condition").loc[order]
            ax.plot(range(len(order)), v.median_attack, marker="o", lw=2, color=ATTACK, label="attack (median)")
            ax.plot(range(len(order)), v.median_bonafide, marker="o", lw=2, color=BONAFIDE, label="bona fide (median)")
            ax.set(xticks=range(len(order)), title=f"{m}: score", ylabel="attack score" if k == 0 else None)
            ax.set_xticklabels(order, rotation=35, ha="right")
            if k == 0:
                ax.legend(frameon=False)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{test}.png", dpi=160)
        plt.close(fig)

    occ = np.load(out / "occlusion.npz")
    maps = [k for k in occ.files if k.startswith("map_")]
    hm = oneclass_root / "patchcore_loose224" / "heatmaps.npz"
    pc = np.load(hm) if hm.exists() else None
    n = len(occ["id"])
    cols = 1 + len(maps) + (pc is not None)
    fig, axes = plt.subplots(n, cols, figsize=(1.9 * cols, 1.9 * n), squeeze=False)
    order = np.lexsort((occ["label"], occ["id"]))  # each twin above its attack
    for r, i in enumerate(order):
        img = occ["images"][i]
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(f"{occ['id'][i]}\n{'attack' if occ['label'][i] else 'bona fide'}", fontsize=8)
        for c, k in enumerate(maps):
            m = occ[k][i]
            lim = max(np.abs(np.stack([occ[k] for k in maps])).max(), 1e-6)
            axes[r, c + 1].imshow(img)
            axes[r, c + 1].imshow(np.kron(m, np.ones((img.shape[0] // m.shape[0],) * 2)), cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.55)
            if r == 0:
                axes[r, c + 1].set_title(f"occlusion: {k[4:]}", fontsize=8)
        if pc is not None:
            j = np.flatnonzero((pc["id"] == occ["id"][i]) & (pc["label"] == occ["label"][i]))
            if len(j):
                h = pc["maps"][j[0]].astype(np.float32)
                axes[r, -1].imshow(img)
                axes[r, -1].imshow(np.kron(h, np.ones((img.shape[0] // h.shape[0],) * 2)), cmap="magma", alpha=0.55,
                                   vmin=np.percentile(pc["maps"], 5), vmax=np.percentile(pc["maps"], 99.5))
            if r == 0:
                axes[r, -1].set_title("PatchCore: patch distance", fontsize=8)
    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("Dev attacks and their bona fide twins. Occlusion: red = hiding this cell lowers the attack score (evidence for attack)", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "occlusion_pairs.png", dpi=140)
    plt.close(fig)
