# screenpad

Screen-replay presentation attack detection (PAD) for face images: give it a folder of images, get back a CSV with an attack score and an attack/bona fide decision for every image. The technical report, submitted alongside this repository, explains the method, the evidence and the trade-offs.

## Scoring your own images

You only need [Docker](https://docs.docker.com/get-docker/). We supply a `Dockerfile` (a recipe), not a pre-built image: step 1 builds the image from it on your machine, and each `docker run` then starts a throwaway container from that image. Run everything from the root of this repository.

**1. Build the image** (once, a few minutes, about 0.8 GB, CPU only):

```bash
docker build -t screenpad .
```

**2. Check the machine and download the weights** (once, about 2 GB, kept in a Docker volume called `screenpad-models`):

```bash
docker run --rm -v screenpad-models:/models screenpad preflight
```

It should end with `preflight: OK`.

**3. Score a folder of images.** Set `DATA` to the folder holding your images and `OUT` to a folder for the results. Both must be absolute paths. Subfolders are scored too, so a folder laid out like the original data (`Bonafide/` and `Screens/` inside it) can be passed as it is.

```bash
DATA=/absolute/path/to/images
OUT=/absolute/path/to/results
docker run --rm -v screenpad-models:/models -v "$DATA":/data/in:ro -v "$OUT":/data/out \
    screenpad infer --input /data/in --output /data/out/scores.csv --recursive
```

The results are written to `$OUT/scores.csv`, one row per file. Inside the container your folders are seen as `/data/in` and `/data/out`, so leave those two paths in the command as they are.

**4. Optional: error rates per subfolder.** If the images are split into folders by class (e.g. `Bonafide/` and `Screens/`), this prints how many in each folder were flagged as attacks. For `Screens/` the rate is the share of attacks caught (1 − APCER); for `Bonafide/` it is the share of genuine images rejected (BPCER).

```bash
docker run --rm -v "$OUT":/data/out --entrypoint python screenpad -c \
  "import pandas as pd; d = pd.read_csv('/data/out/scores.csv'); print(d.groupby(d.image.str.split('/').str[0]).attack.agg(flagged_as_attack='sum', images='count', rate='mean'))"
```

### The output CSV

| Column | Meaning |
|---|---|
| `image` | path relative to the input folder, e.g. `Screens/10744.png` |
| `attack_score` | higher = more likely a photo of a screen. It is a robust z score against genuine calibration images: 0 is a typical genuine image, and each unit is one typical spread of genuine scores |
| `attack` | 1 if `attack_score` >= `threshold`, else 0 |
| `threshold` | 1.231, the score that 10% of genuine calibration images exceed |
| `face_found`, `n_faces` | face detection result; with no face, the image is scored on its largest centred square |
| `z_<member>` | the score of each model in the ensemble, for diagnosis |
| `error` | set when a file cannot be scored (unsupported format, corrupt file); the score is left empty |

A different operating point needs no retraining: apply your own threshold to `attack_score`. A lower threshold catches more attacks and rejects more genuine users.

### Without Docker

[uv](https://docs.astral.sh/uv/) installs the pinned Python (3.12) and the exact locked dependencies (`uv.lock`):

```bash
uv sync --extra torch
uv run --extra torch screenpad preflight
uv run --extra torch python infer.py --input /path/to/images --output scores.csv --recursive
```

## Other useful commands

Shown for Docker; without Docker, replace `docker run --rm -v screenpad-models:/models ... screenpad` with `uv run --extra torch screenpad`.

```bash
# Score with the initial system instead of the deployed one (separate weights folder, downloaded once)
docker run --rm -v screenpad-models:/models -v "$DATA":/data/in:ro -v "$OUT":/data/out \
    screenpad infer --input /data/in --output /data/out/scores_initial.csv --recursive \
    --frozen artefacts/frozen.json --models /models/initial

# List problem files (unsupported, corrupt, very small, 16-bit) before scoring
docker run --rm -v screenpad-models:/models -v "$DATA":/data/in:ro screenpad preflight --input /data/in --recursive

# Every command and its options
docker run --rm screenpad --help
docker run --rm screenpad infer --help
```

Every command also takes:

- `--set key=value` to override any configuration value for that run, e.g. `--set num_threads=4` to use 4 CPU threads instead of 6 (repeatable; the keys are those in `configs/*.yaml`);
- `--config extra.yaml` to merge a whole YAML file of overrides (used by the reproduction commands below, e.g. `screenpad ensemble --config configs/ensemble_photo.yaml`);
- `--device cpu|mps|cuda` (default: the best available; the threshold was calibrated on CPU, which reproduces it exactly).

Each model's own score is already in the `z_<member>` columns of the output, so a single model does not need a separate run.

**Docker notes.** Weights are not baked into the image: the first run downloads them from the public Hugging Face repo [`Jamie1701/screenpad-runs`](https://huggingface.co/Jamie1701/screenpad-runs) into the `screenpad-models` volume, and later runs reuse it (they also work offline, with `--network none`). Tested in fresh containers with 4 CPUs and 4 GB of memory on Linux arm64 and on x86-64 (emulated): the pre-flight golden test passes on both, and every decision matched local scoring (32 development images; 484 external CelebA-Spoof images on arm64). Scores agree across machines to about 0.03 z on average (at most 0.24 z on single images, from floating-point differences between CPU builds), so only images within about 0.25 z of the threshold could change decision between machines.

## Results

No real attack is used for training: a physics-based simulator turns bona fide faces into synthetic screen recaptures, and camera-only controls make the screen the only difference to learn.

| Held-out test (17 attacks, 194 bona fide) | Initial system | **Deployed system** (default) |
|---|---|---|
| Attacks accepted (APCER) | 1/17 | 1/17 |
| Bona fide rejected (BPCER) at the 10% calibration threshold | 5.2% | 9.8% |
| AUC | 0.982 | 0.988 |
| Bona fide rejected under sensor noise (sigma 5) | 66.0% | 18.6% |
| External CelebA-Spoof screen replays caught | 108/108 | 108/108 |

## Inputs and assumptions

- **One face per image**, roughly frontal, as in the supplied data. With several faces the largest is used.
- **Formats:** PNG, JPEG, HEIC/HEIF (the iPhone default), BMP, WebP and TIFF, any case; colour, greyscale, palette, alpha, CMYK and 16-bit images; orientation is applied. All metadata (EXIF, colour profile, file size) is discarded after converting to sRGB, so it cannot act as a shortcut. Any other file gets a row with an error.
- **Screen replays only.** Prints, masks, deepfakes and injection attacks are out of scope.
- **Known weak spots** (report, Sec. 4.7 and 6): very small or heavily blurred faces, heavy JPEG re-compression, and images with a flat band at the crop edge (letterboxing) lean towards "attack". `screenpad preflight --input <folder>` lists small, 16-bit, corrupt and unsupported files before scoring.

## Pre-flight check

`screenpad preflight` (also inside Docker) checks a machine before scoring and exits non-zero on failure:

1. environment and library versions;
2. the ensemble file (hash, members, threshold);
3. the weights (downloaded if missing);
4. the face detector;
5. a golden test: two fixed synthetic images go through the full inference path, and every member's score must match the recorded reference (`artefacts/preflight_reference.json`) within 0.1 z;
6. with `--input`, a scan of the folder for problem files.

## Artefacts

| File | What it is |
|---|---|
| `artefacts/frozen_photo.json` | **deployed system** (default): members H, I (DINOv2-B fine-tuned at 224 and 392 px with photometric augmentation, 3 seeds each) and PatchCore on H's features; calibration statistics; threshold |
| `artefacts/frozen.json` | initial system (A, C, PatchCore on A), pre-registered and scored once on the test set; use with `--frozen artefacts/frozen.json` |
| `artefacts/preflight_reference.json` | golden scores for the pre-flight check |
| `screenpad/assets/face_detection_yunet_2023mar.onnx` | YuNet face detector (bundled, 230 kB) |
| `configs/` | preprocessing, training and ensemble settings |

Weights are fetched on demand (`screenpad fetch-models --dest models`): each run's `config.yaml` and `ckpt/last.pt`, and the PatchCore memory bank.

## Dependencies

Python 3.12; PyTorch (CPU wheels in Docker), timm, OpenCV, Pillow (with pillow-heif for HEIC), NumPy, pandas, scikit-learn, huggingface_hub; versions pinned in `uv.lock`. Inference takes about 1.1 s per image on 6 CPU threads, model loading included.

## Reproducing the work

Every step is a `screenpad` command (`uv run --extra torch screenpad --help`); each takes `--set key.sub=value` overrides and `--config extra.yaml`. The data ([`Jamie1701/screenpad-data`](https://huggingface.co/datasets/Jamie1701/screenpad-data)) and run folders ([`Jamie1701/screenpad-runs`](https://huggingface.co/Jamie1701/screenpad-runs)) live on Hugging Face.

```bash
screenpad pull-data && screenpad manifest && screenpad splits && screenpad audit   # data, splits by identity, leakage checks
screenpad cache                       # aligned face crops (tight 1.0x, loose 1.25x the face box), verified byte-exact
screenpad synth                       # synthetic recaptures + camera-only controls (train/val only)
screenpad train --config configs/runs/H_dinov2b_loose224_photo.yaml        # one fine-tuning run (or scripts/orchestrate.py on Colab)
screenpad oneclass --members patchcore --block 8 --coreset 0.01 --weights-run H_dinov2b_loose224_photo_s42
screenpad ensemble --config configs/ensemble_photo.yaml && screenpad freeze --config configs/ensemble_photo.yaml
screenpad evaluate --split dev        # and --split test --confirm-test, scored once per system
screenpad robustness --split test --suite degradations|edge                 # benign degradations, border test
```

Other commands: `probe` (frozen-feature baselines), `baseline-minifasnet`, `shortcut-test`, `explain`, `bench-inference`, `ssl-adapt`, `mlflow-sync`, `eda`, `synth-gallery`. `notebooks/EDA.ipynb` and `notebooks/Recapture.ipynb` hold the exploratory analysis and the simulator calibration. GPU work ran on Google Colab through `scripts/orchestrate.py` (parallel sessions, progress polling, results verified on the Hub).

```text
screenpad/   data (loading, crops, splits), synth (simulator), train, oneclass, ensemble, eval, explain, infer.py, preflight.py
configs/     default, data, synth, train, ensemble(_photo) and per-run configs
tests/       pytest suite: uv run --extra torch pytest
docs/        Sphinx documentation (overview, model card, CLI and API reference)
```

Documentation site: `uv run --extra torch --group docs sphinx-build docs docs/_build/html`, then open `docs/_build/html/index.html`.

## Third-party components and licences

Only permissively licensed models are used, so nothing here blocks commercial use of the code.

| Component | Use | Licence |
|---|---|---|
| [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (bundled; `screenpad/assets/LICENSE-YuNet`) | face detection | MIT |
| [DINOv2](https://github.com/facebookresearch/dinov2) ViT-S/B/L via [timm](https://github.com/huggingface/pytorch-image-models) | backbone of every shipped member | Apache-2.0 |
| CLIP ViT-B/16, ConvNeXt-T via timm | sweep comparisons only | MIT; Apache-2.0 |
| PyTorch, timm | models, training, inference | BSD-3-Clause; Apache-2.0 |
| [pillow-heif](https://github.com/bigcat88/pillow_heif) | reading HEIC/HEIF inputs | BSD-3-Clause (bundles libheif, LGPL-3.0, dynamically linked) |
| [MiniFASNet](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) @ `b6d5f04` (architecture vendored; weights downloaded and SHA-256 checked) | off-the-shelf comparison only | Apache-2.0 |
| anomalib (EfficientAD), MLflow | one-class experiment; result mirror (dev only) | Apache-2.0 |

YuNet was chosen for its MIT licence, its size (230 kB, CPU, through OpenCV, already a dependency) and its five landmarks; it found a face in 1022 of 1023 images. InsightFace SCRFD/RetinaFace weights and WIDER FACE-trained MTCNN ports were rejected because their weights or training data are research-only. MiniFASNet is the comparison baseline because it is one of very few openly released anti-spoofing models with a permissive licence and weights; its face detector was trained on WIDER FACE, so it stays an evaluation-only baseline.

**Data and weights.** The bona fide images appear to be FFHQ (CC BY-NC-SA 4.0, non-commercial); the screen attacks were provided by the assessors. The published weights were trained on FFHQ-derived images, so they are released for non-commercial research use under the same terms. The code is MIT licensed (`LICENSE`).
