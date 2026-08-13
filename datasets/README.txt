datasets/
=========

This folder contains the dataset configurations and preparation tools used by FractureVision.
The project uses the FracAtlas X-ray dataset and converts it into YOLO segmentation format for
model training and evaluation.

- configs/current/
  Configurations used by the current YOLO11s-Seg pipeline.

- configs/reference/
  Configuration used for positive-only benchmark comparison.

- configs/deprecated/
  Configurations from earlier detection experiments.

- tools/preparation/
  Tools for checking, repairing, preparing, and converting FracAtlas.

- tools/selection/
  Tools for hard-negative mining and training-list generation.

- tools/reference/
  Utilities used for benchmark comparison.

- tools/deprecated/
  Dataset tools from experiments that are no longer part of the current model.

- fracatlas_raw/
  Original downloaded FracAtlas dataset.

- fracatlas_yolo/
  Cleaned intermediate dataset and split metadata.

- fracatlas_seg_yolo/
  Current segmentation dataset used for training and evaluation.

(fracatlas_raw/ | fracatlas_yolo/ | fracatlas_seg_yolo/)
The three dataset directories contain local data and generated files and are excluded from Git.
Only configurations, tools, and documentation are uploaded.