scripts/evaluation/
===================

This folder contains the evaluation and model-selection scripts used for the current FractureVision model.
They measure the performance of the TTA inference strategy and select the checkpoint that provides the best
recall-oriented balance on the validation set.

- evaluate_fracatlas_seg_tta.py
  Evaluates the multi-view TTA and candidate-fusion strategy used during application inference.

- select_fracatlas_seg_recall_f2.py
  Compares saved checkpoints using validation-only F2 criteria and produces the final f2_best.pt model.
