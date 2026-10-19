"""
One-class scorers: model only what bona fide faces look like and score how unusual an image is. No attack, real or synthetic, is used to fit them.
-----------------------------------------------------------------------------------------------------------------------------------------------

They complement the fine-tuned classifiers, which only know the attacks the simulator produces: an attack type the simulator misses can still look unusual. They are ensemble members, not gates: "normal" here means "like the training bona fide" (FFHQ originals and their camera-only controls), so unusual but genuine faces also score high.

Both use a frozen, pretrained DINOv2-base (no fine-tuning), with inputs prepared exactly as for the fine-tuned models (`train.augment.to_input`).

- `PatchCore` (Roth et al., 2022): patch features from an intermediate block, averaged over each patch's 3x3 neighbourhood so every feature sees a little context. A greedy coreset (k-centre) keeps a fraction of all bona fide patch features as the memory bank. An image's score is its worst patch: the largest distance from any of its patches to the nearest bank patch. The per-patch distances form a heatmap of what looked unusual.
- `Mahalanobis`: one feature per image (CLS token and mean patch token of the last block). Fits the bona fide mean and a Ledoit-Wolf (shrinkage) covariance; the score is the squared Mahalanobis distance from that cloud.

Design choices (block, neighbourhood, coreset fraction, input size) are fixed up front, not tuned on the real dev attacks.
"""

from __future__ import annotations

import numpy as np

DINO = "vit_base_patch14_dinov2.lvd142m"


class Dino:
    """Frozen DINOv2-base: patch maps from block `block` and image-level features from the last block."""

    def __init__(self, device, input_size: int = 224, block: int = 8):
        import timm
        import torch

        self.torch, self.device, self.size, self.block = torch, device, input_size, block
        self.model = timm.create_model(DINO, pretrained=True, num_classes=0, img_size=input_size).eval().to(device)
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    def _input(self, images: np.ndarray):
        from ..train.augment import to_input

        x = self.torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float().div_(255)
        return (to_input(x, self.size) - self.mean) / self.std

    def patches(self, images: np.ndarray, batch: int = 32) -> np.ndarray:
        """(N, C, H, W) float16 patch features, 3x3-neighbourhood averaged."""
        F = self.torch.nn.functional
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch):
                (f,) = self.model.forward_intermediates(self._input(images[i : i + batch]), indices=[self.block], norm=False,
                                                        output_fmt="NCHW", intermediates_only=True)
                out.append(F.avg_pool2d(f, 3, stride=1, padding=1, count_include_pad=False).half().cpu().numpy())
        return np.concatenate(out)

    def global_features(self, images: np.ndarray, batch: int = 32) -> np.ndarray:
        """(N, 2C): CLS token and mean patch token of the last block."""
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch):
                t = self.model.forward_features(self._input(images[i : i + batch]))
                out.append(self.torch.cat([t[:, 0], t[:, self.model.num_prefix_tokens :].mean(1)], 1).float().cpu().numpy())
        return np.concatenate(out)


def greedy_coreset(x, n: int, seed: int, proj_dim: int = 128, device=None, chunk: int = 65536):
    """Indices of `n` rows chosen by greedy k-centre (each pick is the row farthest from those already picked), on a random projection for speed."""
    import torch

    g = torch.Generator().manual_seed(seed)
    p = torch.randn(x.shape[1], proj_dim, generator=g) / proj_dim**0.5
    z = torch.cat([x[i : i + chunk].float() @ p for i in range(0, len(x), chunk)]).to(device or x.device)
    picked = [int(torch.randint(len(z), (1,), generator=g))]
    dist = torch.cdist(z, z[picked[-1]][None]).squeeze(1)
    for _ in range(n - 1):
        picked.append(int(dist.argmax()))
        dist = torch.minimum(dist, torch.cdist(z, z[picked[-1]][None]).squeeze(1))
    return np.asarray(picked)


class PatchCore:
    def __init__(self, device, coreset: float = 0.1, seed: int = 0):
        self.device, self.frac, self.seed = device, coreset, seed

    def fit(self, patch_maps: np.ndarray) -> "PatchCore":
        """Memory: the full features stay float16 on the host; only their 128-d projection goes to the device for the coreset."""
        import torch

        n, c, h, w = patch_maps.shape
        x = torch.from_numpy(patch_maps).permute(0, 2, 3, 1).reshape(-1, c)
        idx = greedy_coreset(x, max(1, int(len(x) * self.frac)), self.seed, device=self.device)
        self.bank = x[torch.as_tensor(idx)].float().to(self.device)
        return self

    def save(self, path, meta: dict) -> None:
        """Memory bank (float16) plus the settings needed to rebuild the feature extractor, for inference and explanation."""
        import torch

        torch.save({"bank": self.bank.half().cpu(), **meta}, path)

    @classmethod
    def load(cls, path, device) -> tuple["PatchCore", dict]:
        import torch

        st = torch.load(path, map_location="cpu", weights_only=True)
        pc = cls(device, st["coreset"])
        pc.bank = st.pop("bank").float().to(device)
        return pc, st

    def score(self, patch_maps: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """(image scores, (N, H, W) per-patch nearest-bank distances)."""
        import torch

        n, c, h, w = patch_maps.shape
        x = torch.from_numpy(patch_maps).permute(0, 2, 3, 1).reshape(-1, c).float().to(self.device)
        d = torch.cat([torch.cdist(x[i : i + chunk], self.bank).min(1).values for i in range(0, len(x), chunk)])
        maps = d.reshape(n, h, w).cpu().numpy()
        return maps.reshape(n, -1).max(1), maps


class Mahalanobis:
    def fit(self, feats: np.ndarray) -> "Mahalanobis":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(feats)
        self.mean, self.prec = lw.location_, lw.precision_
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        d = feats - self.mean
        return np.einsum("ij,jk,ik->i", d, self.prec, d)
