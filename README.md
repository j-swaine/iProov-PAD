# screenpad

Screen-replay presentation attack detection (PAD) for face images: reads a folder of images and writes an attack score per image (higher = more likely a screen recapture).

> Work in progress. Inference (`screenpad infer` / `infer.py`) and model artefacts land in later phases; this README is filled in as they do.

## Setup (uv)

[uv](https://docs.astral.sh/uv/) manages the Python version, the virtual environment (`.venv/`) and the locked dependency set (`uv.lock`) in one tool.

```bash
uv sync --extra torch     # create .venv with Python 3.12 + all deps, exactly as locked
uv run screenpad --help   # run anything inside the env without activating it
uv run pytest             # tests
```

Common commands:

| Task | Command |
|---|---|
| Add a dependency | `uv add <pkg>` (or `uv add --optional torch <pkg>`, `uv add --dev <pkg>`) |
| Remove one | `uv remove <pkg>` |
| Activate the env (optional) | `source .venv/bin/activate` |
| Jupyter kernel | select `.venv/bin/python` in VS Code / Jupyter |

## Pipeline

```bash
screenpad pull-data   # download data from the private HF dataset (needs `hf auth login`)
screenpad manifest    # per-image table: id, label, size, ICC/EXIF flags, paired?
screenpad splits      # assign IDs to train/val/test (all real attacks + their twins -> test)
screenpad audit       # leakage checks on the split
```

Every command takes `--set key.sub=value` overrides and `--config extra.yaml` on top of `configs/*.yaml`.

## Data

`DATA/test/data/{Bonafide,Screens}/` locally (not in git); a private copy lives at `Jamie1701/screenpad-data` on the HF Hub for Colab.

## GPU runs (Colab CLI)

`scripts/colab_job.sh` drives a Colab runtime from the terminal (needs `uv tool install google-colab-cli`, a one-off `colab new` login, and `hf auth login`):

```bash
scripts/colab_job.sh up T4            # start a T4 session, upload the working tree, install
scripts/colab_job.sh run pull-data    # VM downloads the data from the HF Hub
scripts/colab_job.sh run splits       # any screenpad command
scripts/colab_job.sh get content/screenpad/results/x.csv ./x.csv
scripts/colab_job.sh down             # always stop the session afterwards
```
