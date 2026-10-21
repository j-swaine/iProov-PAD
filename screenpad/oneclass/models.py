"""
One-class scorers: model only what bona fide faces look like and score how unusual an image is. No attack, real or synthetic, is used to fit them.
--------------------------------------------------------------------------------------------------------------------------------------------------

They complement the fine-tuned classifiers, which only know the attacks the simulator produces: an attack type the simulator misses can still look unusual. They are ensemble members, not gates: "normal" here means "like the training bona fide" (FFHQ originals and their camera-only controls), so unusual but genuine faces also score high.

Both use a frozen, pretrained DINOv2-base (no fine-tuning), with inputs prepared exactly as for the fine-tuned models (`train.augment.to_input`).

- `PatchCore` (Roth et al., 2022): patch features from an intermediate block, averaged over each patch's 3x3 neighbourhood so every feature sees a little context. A greedy coreset (k-centre) keeps a fraction of all bona fide patch features as the memory bank. An image's score is its worst patch: the largest distance from any of its patches to the nearest bank patch. The per-patch distances form a heatmap of what looked unusual.
- `Mahalanobis`: one feature per image (CLS token and mean patch token of the last block). Fits the bona fide mean and a Ledoit-Wolf (shrinkage) covariance; the score is the squared Mahalanobis distance from that cloud.

Design choices (block, neighbourhood, coreset fraction, input size) are never tuned on the real dev attacks. The first configuration (block 8, 224 px, 10% coreset) was fixed up front and scored at chance (synthetic val AUC 0.55, dev 0.46): at that depth DINOv2 features describe the face (who, pose, lighting), and the variation between different people swamps recapture texture. Variants (earlier blocks, 392 px, and features from a fine-tuned classifier) are therefore compared on synthetic val only; dev is reported once for the chosen one.

Optionally (`weights`) the features come from a fine-tuned classifier (A) rather than the pretrained DINOv2. The density model still sees only bona fide, but its feature space was shaped by the simulated attacks, so it is less independent of the simulator than the pretrained variant; reported as such.
"""

from __future__ import annotations

import math

import numpy as np

DINO = "vit_base_patch14_dinov2.lvd142m"
# Feature backbones for the one-class members (all frozen, all ImageNet- or LVD-pretrained, none has seen an attack):
ARCHS = {
    "dinov2_b": DINO,  # Apache-2.0
    "dinov2_l": "vit_large_patch14_dinov2.lvd142m",  # Apache-2.0; 24 blocks
    "wrn50": "wide_resnet50_2.tv_in1k",  # BSD-3 (torchvision weights); PatchCore's published backbone: layers 2 + 3
}
WRN_DIM = 1024  # PatchCore reduces the concatenated layer-2/3 features (512 + 1024 channels) to this many by channel pooling


class Dino:
    """Frozen feature extractor for the one-class members (named for its first backbone).

    DINOv2 (`dinov2_b`, `dinov2_l`): patch maps from block `block`, image-level features from the last block. WideResNet-50 (`wrn50`): PatchCore's original recipe, layer-2 and layer-3 maps (layer 3 upsampled to layer 2's grid, stride 8), concatenated and channel-pooled to 1024; `block` is ignored.
    """

    def __init__(self, device, input_size: int = 224, block: int = 8, weights: str | None = None, arch: str = "dinov2_b"):
        """`weights`: a fine-tuned run's last.pt (DINOv2-base only) instead of the pretrained weights; the attack head is ignored."""
        import timm
        import torch

        self.torch, self.device, self.size, self.block, self.arch = torch, device, input_size, block, arch
        name = ARCHS[arch]
        sd = torch.load(weights, map_location="cpu", weights_only=True) if weights is not None else None
        if arch == "wrn50":
            self.model = timm.create_model(name, pretrained=True, features_only=True, out_indices=(2, 3))
        else:  # a fine-tuned classifier carries a 1-logit head; a self-supervised backbone (oneclass/ssl.py) has none
            self.model = timm.create_model(name, pretrained=sd is None, num_classes=1 if sd is not None and "head.weight" in sd else 0, img_size=input_size)
        if sd is not None:
            pe = self.model.pos_embed
            if sd["pos_embed"].shape != pe.shape:  # trained at another resolution: resample the position grid
                from timm.layers import resample_abs_pos_embed

                g = int(math.sqrt(pe.shape[1] - self.model.num_prefix_tokens))
                sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], (g, g), num_prefix_tokens=self.model.num_prefix_tokens)
            self.model.load_state_dict(sd)
        self.model = self.model.eval().to(device)
        cfg = timm.models.get_pretrained_cfg(name)
        self.mean = torch.tensor(cfg.mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.std, device=device).view(1, 3, 1, 1)

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
                x = self._input(images[i : i + batch])
                if self.arch == "wrn50":
                    f2, f3 = self.model(x)
                    f = self.torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)], 1)
                    f = F.avg_pool2d(f, 3, stride=1, padding=1, count_include_pad=False)
                    b, c, h, w = f.shape
                    f = F.adaptive_avg_pool1d(f.permute(0, 2, 3, 1).reshape(-1, 1, c), WRN_DIM).reshape(b, h, w, WRN_DIM).permute(0, 3, 1, 2)
                else:
                    (f,) = self.model.forward_intermediates(x, indices=[self.block], norm=False, output_fmt="NCHW", intermediates_only=True)
                    f = F.avg_pool2d(f, 3, stride=1, padding=1, count_include_pad=False)
                out.append(f.half().cpu().numpy())
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
