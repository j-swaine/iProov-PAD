"""
EDA figures (static PNGs for the report).

- `pairs_grid`: every real attack beside its bona fide twin, with the crop boxes drawn on. Descriptive only.
- `spectra`: radially averaged log power spectra of tight crops, bona fide vs dev attacks, plus 2-D spectra of the dev pairs. Uses dev attacks only, so nothing here looks at the test attacks.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.io import load_image
from ..data.preprocess import FaceCropper

BONAFIDE_C, ATTACK_C = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the reference palette
CROP_C = {"tight": "#1baf7a", "loose": "#eda100"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})


def _thumb(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    s = size / max(rgb.shape[:2])
    return cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def pairs_grid(splits: pd.DataFrame, root: str | Path, crops: dict[str, float], out: Path, thumb: int = 320) -> None:
    attacks = splits[splits["label"] == 1].sort_values(["split", "id"])
    cropper = FaceCropper()
    n, per_row = len(attacks), 3
    rows = int(np.ceil(n / per_row))
    fig, axes = plt.subplots(rows, per_row * 2, figsize=(per_row * 2 * 2.1, rows * 2.3))
    for ax in axes.flat:
        ax.axis("off")
    for i, (_, r) in enumerate(attacks.iterrows()):
        for j, path in enumerate((r["path"].replace("Screens", "Bonafide"), r["path"])):
            rgb = np.asarray(load_image(Path(root) / path))
            det = cropper.detect(rgb)
            img, s = _thumb(rgb, thumb)
            ax = axes.flat[i * 2 + j]
            ax.imshow(img)
            if det is not None:
                x, y, w, h = det.box
                cx, cy = x + w / 2, y + h / 2
                for name, k in crops.items():
                    side = min(k * max(w, h), min(rgb.shape[:2]))
                    x0 = np.clip(cx - side / 2, 0, rgb.shape[1] - side)
                    y0 = np.clip(cy - side / 2, 0, rgb.shape[0] - side)
                    ax.add_patch(plt.Rectangle((x0 * s, y0 * s), side * s, side * s, fill=False, lw=1.5, ec=CROP_C[name]))
            kind = "bona fide" if j == 0 else "attack"
            ax.set_title(f"{r['id']} {kind} · {r['split']}", fontsize=8, color=INK if j else INK_2)
    handles = [plt.Line2D([], [], color=c, lw=2, label=f"{n} crop (k={crops[n]})") for n, c in CROP_C.items() if n in crops]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Real attacks beside their bona fide twins", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _log_power(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64) / 255.0
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    return np.log10(np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2 + 1e-12)


def _radial(p: np.ndarray) -> np.ndarray:
    h, w = p.shape
    yy, xx = np.indices(p.shape)
    r = np.hypot(yy - h / 2, xx - w / 2).astype(int)
    prof = np.bincount(r.ravel(), p.ravel()) / np.bincount(r.ravel())
    return prof[: h // 2]


def spectra(splits: pd.DataFrame, root: str | Path, k: float, size: int, out: Path, n_bonafide: int = 200, seed: int = 0) -> pd.DataFrame:
    """Radial spectra for train bona fide (sample), dev twins and dev attacks; returns per-image high-frequency energy."""
    cropper = FaceCropper()
    groups = {
        "bona fide (train sample)": splits[(splits["split"] == "train") & (splits["label"] == 0)].sample(n_bonafide, random_state=seed),
        "bona fide (dev twins)": splits[(splits["split"] == "dev") & (splits["label"] == 0)],
        "attack (dev)": splits[(splits["split"] == "dev") & (splits["label"] == 1)],
    }
    profiles, spec2d, hf = {}, {}, []
    for g, df in groups.items():
        profs = []
        for _, r in df.iterrows():
            crop, info = cropper(np.asarray(load_image(Path(root) / r["path"])), k, size)
            p = _log_power(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
            rad = _radial(p)
            profs.append(rad)
            hf.append({"id": r["id"], "group": g, "label": r["label"], "resample_ratio": info.resample_ratio,
                       "hf_minus_lf": float(rad[size // 4 :].mean() - rad[4 : size // 16].mean())})
            if r["split"] == "dev":
                spec2d[(r["id"], r["label"])] = p
        profiles[g] = np.array(profs)
    hf = pd.DataFrame(hf)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    freq = np.arange(size // 2) / size
    styles = {"bona fide (train sample)": (BONAFIDE_C, "-"), "bona fide (dev twins)": (BONAFIDE_C, "--"), "attack (dev)": (ATTACK_C, "-")}
    for g, P in profiles.items():
        c, ls = styles[g]
        m = P.mean(0)
        a1.plot(freq[1:], m[1:], color=c, ls=ls, lw=2, label=f"{g}, n={len(P)}")
        if len(P) > 1:
            a1.fill_between(freq[1:], np.percentile(P, 10, 0)[1:], np.percentile(P, 90, 0)[1:], color=c, alpha=0.12, lw=0)
    a1.set_xlabel("spatial frequency (cycles / pixel)")
    a1.set_ylabel("log10 power")
    a1.set_title(f"Radial power spectrum, tight crop (k={k}) at {size}px", loc="left", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    diff = profiles["attack (dev)"].mean(0) - profiles["bona fide (dev twins)"].mean(0)
    a2.axhline(0, color=INK_2, lw=0.8)
    a2.plot(freq[1:], diff[1:], color=ATTACK_C, lw=2)
    a2.set_xlabel("spatial frequency (cycles / pixel)")
    a2.set_ylabel("attack − twin (log10 power)")
    a2.set_title("Dev attacks minus their own twins", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "spectra_radial.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    ids = sorted({i for i, _ in spec2d})
    fig, axes = plt.subplots(2, len(ids), figsize=(2 * len(ids), 4.2))
    vmin, vmax = np.percentile(np.stack(list(spec2d.values())), [5, 99.9])
    for j, i in enumerate(ids):
        for row, lab in enumerate((0, 1)):
            ax = axes[row, j]
            ax.imshow(spec2d[(i, lab)], cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f"{i} {'attack' if lab else 'bona fide'}", fontsize=8)
    fig.suptitle("2-D log power spectra, dev pairs (off-centre peaks = periodic patterns such as moiré)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "spectra_2d_dev.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return hf
