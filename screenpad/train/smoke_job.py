"""Stand-in job for orchestrator smoke tests: writes a tiny run folder, waits, uploads and verifies it like a real run.

    python -m screenpad.train.smoke_job NAME [SECONDS]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ..config import REPO_ROOT, load_config
from ..utils.provenance import write_provenance
from .hub import ensure_repo, upload_run, verify_run


def main(name: str, seconds: float = 60.0) -> None:
    cfg = load_config()
    run_dir = Path(cfg.train.runs_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    status = run_dir / "STATUS"
    status.write_text(json.dumps({"state": "running", "epoch": 0, "epochs": 1}))
    write_provenance(run_dir / "provenance.json", REPO_ROOT, extra={"run": name, "smoke": True})
    time.sleep(seconds)
    (run_dir / "metrics.csv").write_text("epoch,val_auc\n0,0.5\n")
    ensure_repo(cfg.train.hf_runs_repo)
    upload_run(run_dir, cfg.train.hf_runs_repo, f"runs/{name}")
    problems = verify_run(run_dir, cfg.train.hf_runs_repo, f"runs/{name}")
    status.write_text(json.dumps({"state": "failed", "error": str(problems)} if problems else {"state": "done", "pushed": True}))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 60.0)
