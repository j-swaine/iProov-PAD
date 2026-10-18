"""
CPU thread cap shared by every library the pipeline uses.
"""

from __future__ import annotations

import os
import sys

_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT")
_limiter = None


def limit_threads(n: int) -> int:
    """Cap BLAS/OpenMP pools, joblib workers, OpenCV and torch at `n` threads for the rest of the process.

    The env vars cover child processes and libraries loaded later; threadpoolctl covers BLAS pools already loaded by numpy. sklearn calls should still pass `n_jobs=n` explicitly.
    """
    global _limiter
    for var in _ENV_VARS:
        os.environ[var] = str(n)
    from threadpoolctl import threadpool_limits

    _limiter = threadpool_limits(limits=n)  # kept alive: the limit lasts as long as this object
    try:
        import cv2

        cv2.setNumThreads(n)
    except ImportError:
        pass
    # torch reads OMP_NUM_THREADS when first imported, so only an already-imported torch needs
    # telling; importing it here would add ~2 s to every command.
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(n)
    return n
