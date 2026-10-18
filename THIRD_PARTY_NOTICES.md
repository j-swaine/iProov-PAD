# Third-party components

Components bundled with or downloaded by this repository, and why each was chosen. Licence choice matters because a PAD prototype may be carried into a commercial product: only permissively licensed (MIT / Apache-2.0 / BSD) models are used, so nothing here restricts that.

| Component | Used for | Licence | Source |
|---|---|---|---|
| YuNet face detector (`screenpad/assets/face_detection_yunet_2023mar.onnx`) | face detection for the face-centred crops | MIT ([`screenpad/assets/LICENSE-YuNet`](screenpad/assets/LICENSE-YuNet)) | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| MiniFASNet architecture (`screenpad/baselines/minifasnet_arch.py`, vendored) and weights + Caffe face detector (downloaded at a pinned commit, SHA-256 checked, not bundled) | off-the-shelf comparison baseline only (`screenpad baseline-minifasnet`); not part of the submitted detector | Apache-2.0 | [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) @ `b6d5f04` |

## Why YuNet

YuNet was chosen primarily for its **MIT licence**, which permits commercial use, modification and redistribution with only attribution. Secondary reasons: it is ~230 KB, runs on CPU in a few milliseconds through OpenCV (already a dependency, so the Docker image needs nothing extra), and returns five facial landmarks as well as the box.

Alternatives considered and rejected on licence grounds: InsightFace's SCRFD/RetinaFace detectors (code MIT, but the released model weights are restricted to non-commercial research use) and MTCNN ports trained on WIDER FACE (dataset licence is research-only). MediaPipe (Apache-2.0) was a viable alternative but adds a dependency for no measured gain on this data: YuNet found a face in 1022 of 1023 images.

## Why MiniFASNet as the off-the-shelf comparison

It is one of very few openly released face anti-spoofing models with a permissive licence (Apache-2.0) and published weights, it targets the same print/replay attack family, and it is small enough to run on CPU. It is used purely as an external reference point, run exactly as published, to show what an existing PAD model does on this data. Its face detector was trained on WIDER FACE, whose images are licensed for research use; another reason it stays an evaluation-only baseline rather than a dependency of the detector.

## Data

The bona fide images appear to be FFHQ (CC BY-NC-SA 4.0, non-commercial); the screen attacks are the assessor's. Neither is redistributed: the Hugging Face copy is private.
