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
screenpad pull-data   # download + SHA-256-verify the data tar from the private HF dataset (needs `hf auth login`)
screenpad manifest    # per-image table: id, label, size, ICC/EXIF flags, paired?
screenpad splits      # assign IDs to train/val/dev/test (6 real attacks + twins -> dev, 17 + twins -> test)
screenpad audit       # leakage checks on the split
screenpad eda         # face detections, attack/twin grid, power spectra -> results/eda/
screenpad cache       # aligned face crops (tight k=1.0, loose k=1.25) at 384px -> results/cache/crops_*_aligned.npz,
                      # each verified byte-exact against a fresh recompute (--push uploads to HF)
screenpad synth       # synthetic screen recaptures (+ camera-only bona fide controls) of train/val images,
                      # cropped by the same pipeline -> results/cache/synth_*.npz (never from test)
screenpad pull-data --cache   # fetch the crop caches instead of the raw data
# off-the-shelf comparison (needs OpenCV 4 for its Caffe face detector; layered, project env untouched):
uv run --with 'opencv-python-headless>=4.10,<5' screenpad baseline-minifasnet --splits dev,val
```

Every command takes `--set key.sub=value` overrides and `--config extra.yaml` on top of `configs/*.yaml`.

Commands that need PyTorch (`probe`, `synth-gallery`, training) run with the `torch` extra: `uv run --extra torch screenpad probe`. A plain `uv run` re-syncs the environment to the default dependencies and would uninstall torch again.

```bash
uv run --extra torch screenpad synth-gallery   # contact sheets: results/synth/gallery/
uv run --extra torch screenpad probe           # frozen-backbone linear probes -> results/probes/
```

## EDA

[`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) - class-level distributions and separability of every salient feature (file metadata, face geometry, pixel and spectral statistics), the metadata shortcut demonstration (logistic regression), and what survives cropping. It describes the unaligned crops: run `screenpad eda` and `screenpad cache --set preprocess.align=false` first; re-execute with `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb`.

## Synthetic attacks

Training has no real attacks. Attacks are synthesised from bona fide images by a physical recapture simulator ([`screenpad/synth/recapture.py`](screenpad/synth/recapture.py), ranges in [`configs/synth.yaml`](configs/synth.yaml)). It simulates:
- the display: resolution, tone, gamut and sub-pixel layout;
- glare, reflections and dust;
- the phone's tilt and lens;
- the camera pipeline: Bayer mosaic, noise, demosaicing, denoising, sharpening and JPEG.

Moiré emerges from the sub-pixel grid being sampled by the camera grid, rather than being drawn on. Camera-only bona fide controls go through the camera stage with no screen, so phone processing is not a cue for "attack".

[`notebooks/Recapture.ipynb`](notebooks/Recapture.ipynb) measures what recapture changes on the 6 dev twin pairs only and compares simulated recaptures of those twins with the real attacks. It also documents the calibration and the remaining gaps. Prerequisites: `screenpad cache`, `screenpad synth --splits dev --variants 5 --set synth.camera_only_frac=1.0`.

## Training (fine-tuning on Colab)

`screenpad/train` fine-tunes a pretrained backbone (DINOv2, CLIP or ConvNeXt via timm) to separate real bona fide faces plus camera-only controls from synthetic attacks. Each epoch draws 1:1 classes with every train face in both classes, and GPU augmentation is applied identically to both. The epoch budget is fixed, so the last checkpoint is the default; the best checkpoint by synthetic val AUC is also saved. Real dev attacks are logged on every evaluation but never used for selection.

Every evaluation reports two shortcut checks:
- `ratio_spearman_bonafide` / `ratio_spearman_attack`: rank correlation between score and crop downsample factor within each class (near 0 is good, so resampling history does not drive the score).
- `camera_vs_original_auc`: AUC separating camera-only bona fide from original bona fide (0.5 is good, so phone processing alone does not look like an attack).

```bash
uv run --extra torch screenpad train --config configs/runs/A_dinov2b_loose224.yaml          # one run, locally
uv run --extra torch screenpad train --config configs/runs/A_dinov2b_loose224.yaml --bench  # batch-size probe only
uv run --extra torch python scripts/orchestrate.py configs/runs/sweep.yaml --slots 2       # queue on parallel Colab L4 sessions
```

Each run writes `results/runs/<name>/` containing the resolved `config.yaml`, `provenance.json`, `bench.json` (if probed), `metrics.csv` (per epoch), `scores_best.csv` and `scores_last.csv` (per image, val and dev), `summary.json`, `ckpt/best.pt` and `ckpt/last.pt`, `train.log` and a `STATUS` file. With `--push`, the folder is uploaded to the private HF model repo `Jamie1701/screenpad-runs` under `runs/<name>/`, together with a `MANIFEST.json` of every file's byte size and SHA-256. The orchestrator independently checks the Hub listing against that manifest before a job counts as done.

The orchestrator runs one thread per Colab session (`pad-q1`, `pad-q2`, ...). Sessions that Colab refuses (session limit or capacity) are retried while jobs remain, and a failed job is retried once. Every Colab CLI call goes through one lock, because the CLI hung when two sessions ran `colab exec` at the same moment. State is kept in `results/orchestrator/<queue>/state.json` and `events.log`, and re-running the script skips finished jobs. Each session is stopped when its thread exits. In testing on 2026-10-18 the account was granted 2 concurrent L4 sessions.

## Experiment tracking (MLflow)

Every run folder (config, provenance, per-epoch metrics, scores) is the record of truth, on the private HF runs repo. `screenpad mlflow-sync` mirrors them, plus the one-class members, the shrink test and the ensemble ablation, into a local MLflow store for side-by-side comparison. Training itself has no MLflow dependency, so Colab VMs need no tracking server. Re-running the sync replaces each mirrored run.

```bash
uv run --extra torch screenpad mlflow-sync
uv run --extra torch mlflow ui --backend-store-uri sqlite:///results/mlflow.db --port 5001 --workers 1   # http://127.0.0.1:5001; Ctrl+C to stop
# if the port stays busy after stopping: kill $(lsof -a -t -iTCP:5001 -sTCP:LISTEN -c Python)
```

## Data

`DATA/test/data/{Bonafide,Screens}/` locally (not in git); a private copy lives at `Jamie1701/screenpad-data` on the HF Hub as one byte-identical tar plus per-file SHA-256s (`scripts/push_data_to_hf.py` builds it; `screenpad pull-data` downloads, extracts and verifies it).

## GPU runs (Colab CLI)

`scripts/colab_job.sh` drives a Colab runtime from the terminal (needs `uv tool install google-colab-cli`, a one-off `colab new` login, and `hf auth login`):

```bash
scripts/colab_job.sh up T4            # start a T4 session, upload the working tree, install
scripts/colab_job.sh run pull-data    # VM downloads the data from the HF Hub
scripts/colab_job.sh run splits       # any screenpad command
scripts/colab_job.sh get content/screenpad/results/x.csv ./x.csv
scripts/colab_job.sh down             # always stop the session afterwards
```

## Third-party components

Face detection uses OpenCV's YuNet model, bundled in `screenpad/assets/`. It was chosen for its **MIT licence** (commercial use permitted); see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for every bundled model, its licence, and the alternatives rejected on licence grounds.
