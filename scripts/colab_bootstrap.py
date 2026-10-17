"""Runs inside the Colab kernel (sent by `colab_job.sh push`): unpack the code tarball, install the package, move the HF token into place.

The kernel is long-lived, so the working directory and environment set here persist for later `colab_job.sh run` calls in the same session.
"""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

REPO = Path("/content/screenpad")
TARBALL = Path("/content/screenpad.tar.gz")
TOKEN = Path("/content/.hf_token")

shutil.rmtree(REPO, ignore_errors=True)
REPO.mkdir(parents=True)
with tarfile.open(TARBALL) as t:
    t.extractall(REPO, filter="data")
TARBALL.unlink()

if TOKEN.exists():
    dest = Path.home() / ".cache" / "huggingface" / "token"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(TOKEN, dest)
    dest.chmod(0o600)

os.chdir(REPO)
os.environ["SCREENPAD_DATA_ROOT"] = "/content/DATA/test/data"
r = subprocess.run(["pip", "install", "-q", "-e", ".[torch]"], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:], r.stderr[-3000:])
    raise RuntimeError("pip install failed")

print("code version:", (REPO / "SOURCE_VERSION").read_text().strip())
import torch

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("__SCREENPAD_" + "STEP_OK__")  # split so a traceback quoting this line cannot fake success
