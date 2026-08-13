scripts/training/
=================

This folder contains the training scripts used to develop the current FractureVision YOLO11s-Seg model.
The training process is divided into three stages to establish fracture localization, reduce false positives,
and improve recall.

- train_fracatlas_seg_positive.py
  Trains the initial segmentation model using fracture-positive images.

- train_fracatlas_seg_final.py
  Fine-tunes the model with positive images and selected hard negatives.

- train_fracatlas_seg_recall_v2.py
  Performs recall-oriented fine-tuning to detect more potential fracture
  regions while retaining reasonable precision.
