# FractureVision

FractureVision is a desktop application for localizing suspected fracture
regions in X-ray images. It combines a YOLO11s-Seg model with multi-view
test-time augmentation, candidate fusion, duplicate suppression, and an
interactive PyQt5 interface.

> FractureVision is an educational and portfolio project. It is not a medical
> device and must not be used for clinical diagnosis or treatment decisions.

## Features

- Imports common X-ray image formats through a desktop interface.
- Localizes and segments multiple suspected fracture regions.
- Displays each candidate separately to avoid overlapping annotations.
- Reports an Evidence Score, Raw Fusion score, and multi-view support.
- Preserves the original image and exports annotated candidates with JSON data.
- Supports Apple Silicon MPS, CUDA, and CPU inference.

## Model Pipeline

The current model is a single-class YOLO11s-Seg model trained on FracAtlas. The
development pipeline includes positive-only pretraining, hard-negative
fine-tuning, recall-oriented fine-tuning, and validation-only F2 checkpoint
selection.

Inference combines three views of the same image:

1. Original image
2. Horizontally flipped image
3. Mild contrast-enhanced image

Nearby results are fused and visually duplicated regions are suppressed before
they are displayed.

## Project Structure

```text
FractureVision/
├── app/                 Desktop interface and inference engine
├── datasets/            Dataset configuration and preparation tools
├── scripts/
│   ├── training/        Current model-training pipeline
│   ├── evaluation/      TTA evaluation and F2 checkpoint selection
│   └── deprecated/      Superseded experiments kept for reference
├── ultralytics/         Bundled Ultralytics framework source
├── tests/               Framework tests
├── requirements.txt
└── pyproject.toml
```

## Installation

Python 3.10–3.12 is recommended.

```bash
git clone git@github.com:hwspen7/FractureVision.git
cd FractureVision

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Model Checkpoint

Model weight files are not stored in Git history. Place the selected checkpoint
at:

```text
runs/segment/fracatlas_yolo11s_seg_recall_v2/weights/f2_best.pt
```

The release checkpoint can be distributed separately through a GitHub Release.

## Run the Desktop Application

```bash
PYTHONPATH=. python app/fracture_detection_app.py
```

## Run Command-Line Inference

```bash
PYTHONPATH=. python app/fracture_inference.py \
  --input path/to/xray.jpg \
  --output inference_outputs/example \
  --device cpu
```

Use `--device mps` on a compatible Apple Silicon Mac or `--device 0` for the
first CUDA GPU.

## Dataset

This project uses FracAtlas. Dataset images and generated labels are not included
in this repository. See `datasets/README.txt` for the preparation workflow and
directory structure.

## Result Interpretation

The interface reports two model-derived scores:

- **Evidence Score**: a validation-threshold-normalized score used to make
  candidate ranking easier to interpret in the interface.
- **Raw Fusion**: the fused confidence produced by the multi-view inference
  pipeline.

Neither score is a calibrated medical probability.

## License and Attribution

This repository includes modified Ultralytics source code and retains the
AGPL-3.0 license in `LICENSE`. FracAtlas must be downloaded and used under its
own applicable license and terms.
