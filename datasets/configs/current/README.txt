datasets/configs/current/
=========================

This folder contains the dataset configurations used by the current FractureVision YOLO11s-Seg
training and evaluation pipeline.

- fracatlas_seg_positive.yaml
  Positive-only configuration used for initial segmentation training.

- fracatlas_seg_mixed.yaml
  Uses positive training data with the complete validation and test splits.

- fracatlas_seg_final.yaml
  Configuration used for mixed positive and hard-negative fine-tuning.

- fracatlas_seg_recall_v2.yaml
  Final recall-oriented configuration used for training and evaluation.