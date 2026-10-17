"""Entry point for ``python -m screenpad``; delegates straight to `cli.main`."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
