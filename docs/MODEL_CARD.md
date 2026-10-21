# Model card: screenpad deployed system (screen-replay PAD for face images)

Research prototype. Format after Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019).

## Model details

- **Task:** presentation attack detection (PAD) for a single face image: bona fide (a live person) vs a face shown on a screen and photographed (screen replay). Output: an attack score per image (robust z against bona fide calibration images) and a decision at a calibrated threshold.
- **Pipeline:** metadata-stripping loader (EXIF orientation, sRGB, metadata discarded); YuNet face detection (MIT); eye-aligned square crop of 1.25 x the face box at 384 px; ensemble of three members, each calibrated on bona fide images only, averaged with equal weight.
- **Members (deployed system, `artefacts/frozen_photo.json`):** H and I, DINOv2-B fine-tuned at 224 and 392 px with photometric augmentation (blur, noise, gamma, brightness, applied identically to both classes), 3 seeds each; PatchCore on H's features (density model of bona fide patches).
- **Training:** synthetic attacks from a physics-based recapture simulator, plus camera-only bona fide controls; no real attack used. 20 epochs, AdamW, layer-wise learning-rate decay, cosine schedule, bf16 on Colab L4 GPUs.
- **Initial system** (`artefacts/frozen.json`): A + C (the same recipes without photometric augmentation) + PatchCore on A; pre-registered and scored once on the test set; kept for comparison.
- **Versioning:** code in git; run folders (config, provenance, metrics, per-image scores, weights) on the public Hugging Face repo `Jamie1701/screenpad-runs`; ensemble files in `artefacts/`.

## Intended use

- Research and evaluation of a screen-replay PAD component in a security-first setting (e.g. remote identity verification for financial services), where a rejection routes the user to a step-up check rather than a lock-out.
- Inputs like the development data: single, roughly frontal face images.

## Out of scope

- Other attack instruments: prints, masks, 3D, deepfakes, injection attacks.
- Any standalone decision about a person; identification; surveillance.
- Commercial use of the weights (trained on FFHQ-derived images, CC BY-NC-SA 4.0).

## Data

- Bona fide: 1000 FFHQ faces (1024 px). Attacks: 23 phone recaptures (iPhone 15 x14, Galaxy A12 x9), provided for the assignment.
- Splits by identity: train (680 bona fide), validation (120), dev (6 attacks + twins), test (17 attacks + twins + 177 bona fide).
- Calibration: 240 validation bona fide images (120 originals + 120 camera-only controls).

## Metrics

APCER (attacks accepted) and BPCER (bona fide rejected) at the threshold with exact Clopper-Pearson 95% intervals; AUC with stratified bootstrap intervals; BPCER@APCER0; per-phone APCER; BPCER per skin-tone band (ITA, a pixel proxy). ISO/IEC 30107-3 vocabulary.

## Results (held-out test: 17 attacks, 194 bona fide)

| | Initial system | Deployed system |
|---|---|---|
| APCER | 1/17 = 5.9% [0.1, 28.7] | 1/17 = 5.9% [0.1, 28.7] |
| BPCER at the 10% calibration threshold | 10/194 = 5.2% [2.5, 9.3] | 19/194 = 9.8% [6.0, 14.9] |
| AUC | 0.982 [0.957, 0.999] | 0.988 [0.971, 0.999] |
| BPCER@APCER0 | 18.6% | 12.4% |
| BPCER, noise sigma 5 / blur sigma 2 / 4x downscale | 66.0% / 41.2% / 42.3% | 18.6% / 31.4% / 33.0% |
| Attack misses summed over 14 benign degradations | 13 | 8 |
| CelebA-Spoof PC-screen replays (external, no retraining) | 0/108 missed | 0/108 missed |

Trade-off: the deployed system ranks better and is far more robust to degraded genuine captures, but rejects more clean bona fide images at the shared 10% rule (its rate lands on the 10% target; the initial system's fell below it). The threshold is tunable on bona fide data. The deployed system was chosen on validation evidence alone, before its own test run. Sources: `results/evaluation/`; report Sec. 4.7 and 5.
