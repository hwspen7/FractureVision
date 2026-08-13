datasets/tools/selection/
=========================

This folder contains the data-selection tools used by the current segmentation pipeline.

- mine_fracatlas_seg_hard_negatives.py
  Finds negative images that the model incorrectly considers likely fractures.

- create_fracatlas_seg_final_list.py
  Creates the mixed positive and hard-negative fine-tuning list.

- create_fracatlas_seg_recall_list.py
  Creates the recall-oriented training list used during the final training stage.