"""
Device utilities for PyTorch.
"""


def resolve_device(prefer: str = "auto") -> "object":
    """Return a ``torch.device``. ``prefer`` in {auto, cuda, cpu, mps}, or
    already a ``torch.device`` (returned unchanged, so this is safe to call
    on a value that may or may not have been resolved yet)."""
    import torch

    if isinstance(prefer, torch.device):
        return prefer
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if (
        prefer in ("auto", "mps")
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")
