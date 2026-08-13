datasets/tools/preparation/
===========================

This folder contains the tools used to validate and prepare FracAtlas before training the segmentation model.

- check_fracatlas_dataset.py
  Checks image and label consistency across the dataset splits.

- prepare_fracatlas_dataset.py
  Removes duplicate content and creates the cleaned intermediate dataset.

- repair_fracatlas_jpegs.py
  Repairs incompatible JPEG files in the prepared dataset copy.

- prepare_fracatlas_segmentation.py
  Converts the FracAtlas segmentation annotations into YOLO segmentation format.