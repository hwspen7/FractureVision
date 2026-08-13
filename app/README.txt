app/
===============
This folder contains the FractureVision desktop application.
Users can import X-ray images, run fracture analysis, review detected regions, and save the results.

- fracture_detection_app.py
  Provides the PyQt5 interface for loading images, viewing results, and exporting analysis files.

- fracture_inference.py
  Handles YOLO11s-Seg inference, combines detection results, calculates evidence scores, and generates annotated images.
