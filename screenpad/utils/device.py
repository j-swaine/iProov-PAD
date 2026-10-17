"""
Device utilities for PyTorch.
"""

import os


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


def resolve_num_workers(configured=None) -> int:
    """Return the number of DataLoader worker processes to use.

    An explicit value is honoured directly. None auto-detects as ``usable_cpus - 1``, leaving one core free for the main process. ``os.sched_getaffinity(0)`` respects container/cgroup CPU limits (Linux only, absent on macOS) so it is preferred where available.
    """
    if configured is not None:
        return int(configured)
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count() or 2
    return max(1, n - 1)
