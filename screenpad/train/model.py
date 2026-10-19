"""
Fine-tuning models: a pretrained timm backbone with a single-logit head (the attack logit).

The whole backbone is trained, with layer-wise learning-rate decay: the head and last block get the full rate and each earlier block `layer_decay` times the next one's, so early, general-purpose layers move little (the standard recipe for fine-tuning pretrained ViTs and ConvNeXts).
"""

from __future__ import annotations

BACKBONES = {
    # name: (timm id, licence, whether it takes an img_size argument)
    "dinov2_b": ("vit_base_patch14_dinov2.lvd142m", "Apache-2.0", True),
    "dinov2_s": ("vit_small_patch14_dinov2.lvd142m", "Apache-2.0", True),
    "clip_b16": ("vit_base_patch16_clip_224.openai", "MIT", True),
    "convnext_t": ("convnext_tiny.fb_in22k_ft_in1k", "Apache-2.0", False),
}


def build_model(name: str, input_size: int, pretrained: bool = True):
    import timm

    model_id, _, takes_size = BACKBONES[name]
    kw = {"img_size": input_size} if takes_size else {}
    return timm.create_model(model_id, pretrained=pretrained, num_classes=1, **kw)


def build_optimizer(model, lr: float, weight_decay: float, layer_decay: float):
    from timm.optim import create_optimizer_v2

    return create_optimizer_v2(model, opt="adamw", lr=lr, weight_decay=weight_decay, layer_decay=layer_decay)


def normalisation(model) -> tuple[tuple, tuple]:
    cfg = model.pretrained_cfg
    return tuple(cfg["mean"]), tuple(cfg["std"])
