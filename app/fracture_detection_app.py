import json
import sys
from pathlib import Path

import cv2
from PyQt5.QtCore import QThread, QTimer, QUrl, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fracture_inference import (
    FractureInferenceEngine,
    read_image,
    resize_radiograph_for_display,
    save_png,
    serializable_detection,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT
    / "experiments"
    / "segmentation"
    / "training"
    / "recall_optimized"
    / "weights"
    / "f2_best.pt"
)


class ImagePanel(QLabel):
    def __init__(self, placeholder, parent=None):
        super().__init__(parent)
        self.placeholder = placeholder
        self.source_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(500, 500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText(placeholder)
        self.setObjectName("imagePanel")

    def set_bgr_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        self.source_pixmap = QPixmap.fromImage(qimage)
        self._refresh_pixmap()

    def clear_image(self):
        self.source_pixmap = None
        self.clear()
        self.setText(self.placeholder)

    def _refresh_pixmap(self):
        if self.source_pixmap is None:
            return
        self.setPixmap(
            self.source_pixmap.scaled(
                self.contentsRect().size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()


class InferenceThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, engine, image_path, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.image_path = image_path

    def run(self):
        try:
            result = self.engine.predict(self.image_path)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.succeeded.emit(result)


class FractureDetectionWindow(QMainWindow):
    LEVEL_LABELS = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "trace": "Trace",
    }

    def __init__(self):
        super().__init__()
        self.engine = None
        self.image_path = None
        self.original_image = None
        self.prediction = None
        self.inference_thread = None
        self.current_candidate_index = 0
        self.show_full_candidate_result = False
        self.last_saved_directory = None

        self._build_ui()
        self._apply_style()
        QTimer.singleShot(0, self._initialize_model)

    def _build_ui(self):
        self.setWindowTitle("Fracture Detection and Localization System")
        self.resize(1560, 980)
        self.setMinimumSize(1200, 780)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 16, 20, 16)
        root_layout.setSpacing(10)
        self.setCentralWidget(central)

        title_row = QHBoxLayout()
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel("Fracture Detection and Localization System")
        title.setObjectName("title")
        subtitle = QLabel("YOLO11s-Seg | Single-Model Three-View Fusion | FracAtlas Demo")
        subtitle.setObjectName("subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        self.model_badge = QLabel("Loading model...")
        self.model_badge.setObjectName("modelBadge")
        self.model_badge.setAlignment(Qt.AlignCenter)
        title_row.addLayout(title_block)
        title_row.addStretch()
        title_row.addWidget(self.model_badge)
        root_layout.addLayout(title_row)

        warning = QLabel(
            "For project demonstration only. Not for medical diagnosis. "
            "Evidence Scores are validation-threshold normalized, not fracture probabilities."
        )
        warning.setObjectName("warning")
        warning.setWordWrap(True)
        root_layout.addWidget(warning)

        image_titles = QHBoxLayout()
        original_title = QLabel("Original X-ray")
        result_title = QLabel("Current Candidate Focus View")
        original_title.setObjectName("sectionTitle")
        result_title.setObjectName("sectionTitle")
        image_titles.addWidget(original_title, 1)
        image_titles.addWidget(result_title, 1)
        root_layout.addLayout(image_titles)

        image_splitter = QSplitter(Qt.Horizontal)
        image_splitter.setChildrenCollapsible(False)
        self.original_panel = ImagePanel("Select an X-ray image")
        self.result_panel = ImagePanel("Waiting for analysis")
        image_splitter.addWidget(self.original_panel)
        image_splitter.addWidget(self.result_panel)
        image_splitter.setSizes([750, 750])
        root_layout.addWidget(image_splitter, 1)

        navigation = QHBoxLayout()
        navigation.setSpacing(10)
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.view_mode_button = QPushButton("Show Full Result")
        self.previous_button.setObjectName("navigationButton")
        self.next_button.setObjectName("navigationButton")
        self.view_mode_button.setObjectName("navigationButton")
        self.page_label = QLabel("Not analyzed")
        self.page_label.setObjectName("pageBadge")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.current_detail_label = QLabel(
            "Each independent candidate is displayed separately to prevent overlap"
        )
        self.current_detail_label.setObjectName("currentDetail")

        self.previous_button.clicked.connect(self._show_previous_candidate)
        self.next_button.clicked.connect(self._show_next_candidate)
        self.view_mode_button.clicked.connect(self._toggle_candidate_view)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.view_mode_button.setEnabled(False)

        navigation.addStretch()
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.page_label)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.view_mode_button)
        navigation.addSpacing(12)
        navigation.addWidget(self.current_detail_label)
        navigation.addStretch()
        root_layout.addLayout(navigation)

        details_frame = QFrame()
        details_frame.setObjectName("detailsFrame")
        details_layout = QVBoxLayout(details_frame)
        details_layout.setContentsMargins(14, 10, 14, 10)
        details_layout.setSpacing(7)

        details_header = QHBoxLayout()
        result_heading = QLabel("Candidate List")
        result_heading.setObjectName("sectionTitle")
        self.summary_label = QLabel("Not analyzed")
        self.summary_label.setObjectName("summary")
        details_header.addWidget(result_heading)
        details_header.addStretch()
        details_header.addWidget(self.summary_label)
        details_layout.addLayout(details_header)

        self.result_table = QTableWidget(0, 6)
        self.result_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Evidence Score",
                "Raw Fusion",
                "Level",
                "View Support",
                "Bounding Box",
            ]
        )
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setSelectionMode(QTableWidget.SingleSelection)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.cellClicked.connect(self._jump_to_candidate)
        self.result_table.setMaximumHeight(150)
        details_layout.addWidget(self.result_table)

        candidate_note = QLabel(
            "Click a row to open that candidate. Trace and low-evidence candidates may be model noise. "
            "Candidates supported by multiple views are generally more stable."
        )
        candidate_note.setObjectName("candidateNote")
        candidate_note.setWordWrap(True)
        details_layout.addWidget(candidate_note)
        root_layout.addWidget(details_frame)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.open_button = QPushButton("Select Image")
        self.analyze_button = QPushButton("Run Analysis")
        self.save_button = QPushButton("Save Result Set")
        self.clear_button = QPushButton("Clear")
        self.open_button.setObjectName("secondaryButton")
        self.analyze_button.setObjectName("primaryButton")
        self.save_button.setObjectName("secondaryButton")
        self.clear_button.setObjectName("quietButton")
        self.open_button.clicked.connect(self._open_image)
        self.analyze_button.clicked.connect(self._start_analysis)
        self.save_button.clicked.connect(self._save_result)
        self.clear_button.clicked.connect(self._clear)
        self.analyze_button.setEnabled(False)
        self.save_button.setEnabled(False)
        action_row.addWidget(self.open_button)
        action_row.addWidget(self.analyze_button)
        action_row.addWidget(self.save_button)
        action_row.addWidget(self.clear_button)
        action_row.addStretch()
        root_layout.addLayout(action_row)

        self.status_label = QLabel("Initializing...")
        self.status_label.setObjectName("status")
        root_layout.addWidget(self.status_label)

        self._build_save_toast(central)

    def _build_save_toast(self, parent):
        self.save_toast = QFrame(parent)
        self.save_toast.setObjectName("saveToast")
        self.save_toast.setFixedSize(440, 158)

        shadow = QGraphicsDropShadowEffect(self.save_toast)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(15, 23, 42, 90))
        self.save_toast.setGraphicsEffect(shadow)

        toast_layout = QVBoxLayout(self.save_toast)
        toast_layout.setContentsMargins(20, 16, 20, 14)
        toast_layout.setSpacing(8)

        toast_header = QHBoxLayout()
        toast_badge = QLabel("SAVED")
        toast_badge.setObjectName("toastBadge")
        toast_title = QLabel("Result set saved")
        toast_title.setObjectName("toastTitle")
        toast_header.addWidget(toast_badge)
        toast_header.addWidget(toast_title)
        toast_header.addStretch()
        toast_layout.addLayout(toast_header)

        self.toast_message = QLabel("The result files are ready.")
        self.toast_message.setObjectName("toastMessage")
        self.toast_message.setWordWrap(True)
        toast_layout.addWidget(self.toast_message)

        toast_actions = QHBoxLayout()
        toast_actions.addStretch()
        self.open_result_folder_button = QPushButton("Open Folder")
        self.dismiss_toast_button = QPushButton("Dismiss")
        self.open_result_folder_button.setObjectName("toastPrimaryButton")
        self.dismiss_toast_button.setObjectName("toastSecondaryButton")
        self.open_result_folder_button.clicked.connect(self._open_saved_directory)
        self.dismiss_toast_button.clicked.connect(self._hide_save_toast)
        toast_actions.addWidget(self.dismiss_toast_button)
        toast_actions.addWidget(self.open_result_folder_button)
        toast_layout.addLayout(toast_actions)

        self.toast_timer = QTimer(self)
        self.toast_timer.setSingleShot(True)
        self.toast_timer.timeout.connect(self._hide_save_toast)
        self.save_toast.hide()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #F8FAFC;
                color: #1E293B;
                font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
                font-size: 14px;
            }
            QLabel#title { font-size: 26px; font-weight: 700; color: #0F172A; }
            QLabel#subtitle { color: #64748B; font-size: 13px; }
            QLabel#modelBadge {
                background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE;
                border-radius: 16px; padding: 7px 14px; font-weight: 600;
            }
            QLabel#warning {
                background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 8px;
                color: #9A3412; padding: 8px 12px;
            }
            QLabel#sectionTitle { font-size: 15px; font-weight: 700; color: #0F172A; padding: 1px 4px; }
            QLabel#imagePanel {
                background: #0F172A; color: #94A3B8; border: 1px solid #CBD5E1;
                border-radius: 10px; padding: 6px;
            }
            QFrame#detailsFrame {
                background: white; border: 1px solid #E2E8F0; border-radius: 10px;
            }
            QLabel#summary { color: #475569; font-weight: 600; padding: 4px 8px; }
            QLabel#candidateNote { color: #64748B; font-size: 12px; }
            QLabel#status { color: #475569; padding: 2px 0; }
            QLabel#pageBadge {
                min-width: 92px; background: #EEF2FF; color: #4338CA;
                border: 1px solid #C7D2FE; border-radius: 8px; padding: 7px 12px;
                font-weight: 700;
            }
            QLabel#currentDetail { color: #475569; font-weight: 600; }
            QTableWidget {
                background: white; alternate-background-color: #F8FAFC;
                border: 1px solid #E2E8F0; border-radius: 7px; gridline-color: #F1F5F9;
                selection-background-color: #EEF2FF; selection-color: #312E81;
            }
            QHeaderView::section {
                background: #F1F5F9; color: #334155; border: none;
                border-bottom: 1px solid #CBD5E1; padding: 6px 5px; font-weight: 600;
            }
            QPushButton {
                min-width: 100px; min-height: 36px; border-radius: 8px;
                padding: 3px 14px; font-weight: 600;
            }
            QPushButton#primaryButton { background: #4F46E5; color: white; border: 1px solid #4F46E5; }
            QPushButton#primaryButton:hover { background: #4338CA; }
            QPushButton#secondaryButton, QPushButton#navigationButton {
                background: white; color: #1E293B; border: 1px solid #CBD5E1;
            }
            QPushButton#secondaryButton:hover, QPushButton#navigationButton:hover { background: #F1F5F9; }
            QPushButton#quietButton { background: transparent; color: #64748B; border: 1px solid transparent; }
            QPushButton:disabled { background: #E2E8F0; color: #94A3B8; border-color: #E2E8F0; }
            QFrame#saveToast {
                background: #0F172A;
                border: 1px solid #334155;
                border-radius: 14px;
            }
            QLabel#toastBadge {
                background: #10B981;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#toastTitle {
                background: transparent;
                color: #F8FAFC;
                border: none;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#toastMessage {
                background: transparent;
                color: #CBD5E1;
                border: none;
                font-size: 13px;
            }
            QPushButton#toastPrimaryButton, QPushButton#toastSecondaryButton {
                min-width: 88px;
                min-height: 30px;
                border-radius: 7px;
                padding: 2px 12px;
            }
            QPushButton#toastPrimaryButton {
                background: #4F46E5;
                color: white;
                border: 1px solid #6366F1;
            }
            QPushButton#toastPrimaryButton:hover { background: #6366F1; }
            QPushButton#toastSecondaryButton {
                background: #1E293B;
                color: #CBD5E1;
                border: 1px solid #475569;
            }
            QPushButton#toastSecondaryButton:hover { background: #334155; }
            """
        )

    def _initialize_model(self):
        try:
            self.engine = FractureInferenceEngine(MODEL_PATH)
        except Exception as error:
            self.model_badge.setText("Model Load Failed")
            self.model_badge.setStyleSheet("background:#FEE2E2;color:#991B1B;border-color:#FECACA;")
            self.status_label.setText(f"Model load failed: {error}")
            QMessageBox.critical(self, "Model Load Failed", f"Could not load the model:\n{error}")
            return

        self.model_badge.setText(f"Model Ready | {self.engine.device}")
        self.status_label.setText(
            "Select an X-ray image. Fused candidates and normalized Evidence Scores "
            "will be displayed individually."
        )
        self.analyze_button.setEnabled(self.image_path is not None)

    def _open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select an X-ray Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not file_name:
            return
        try:
            image = read_image(file_name)
        except Exception as error:
            QMessageBox.warning(self, "Image Read Failed", str(error))
            return

        self.image_path = Path(file_name).resolve()
        self.original_image = image
        self.prediction = None
        self.current_candidate_index = 0
        self.show_full_candidate_result = False
        self.original_panel.set_bgr_image(resize_radiograph_for_display(image, 1200))
        self.result_panel.clear_image()
        self.result_table.setRowCount(0)
        self.summary_label.setText("Image loaded; waiting for analysis")
        self.page_label.setText("Not analyzed")
        self.current_detail_label.setText(
            "Each independent candidate is displayed separately to prevent overlap"
        )
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.view_mode_button.setEnabled(False)
        self.status_label.setText(f"Loaded: {self.image_path.name}")
        self.analyze_button.setEnabled(self.engine is not None)
        self.save_button.setEnabled(False)

    def _set_busy(self, busy):
        self.open_button.setEnabled(not busy)
        self.analyze_button.setEnabled(not busy and self.engine is not None and self.image_path is not None)
        self.save_button.setEnabled(not busy and self.prediction is not None)
        self.clear_button.setEnabled(not busy)
        self.result_table.setEnabled(not busy)
        self.analyze_button.setText("Analyzing..." if busy else "Run Analysis")
        if busy:
            self.previous_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.view_mode_button.setEnabled(False)
        else:
            self._refresh_navigation_buttons()

    def _start_analysis(self):
        if self.engine is None:
            QMessageBox.information(self, "Model Not Ready", "The model has not loaded successfully.")
            return
        if self.image_path is None:
            QMessageBox.information(self, "No Image Selected", "Select an image first.")
            return
        if self.inference_thread is not None and self.inference_thread.isRunning():
            return

        self._set_busy(True)
        self.status_label.setText(
            "Running original, horizontal-flip, and mild-contrast view fusion..."
        )
        self.inference_thread = InferenceThread(self.engine, self.image_path, self)
        self.inference_thread.succeeded.connect(self._analysis_succeeded)
        self.inference_thread.failed.connect(self._analysis_failed)
        self.inference_thread.finished.connect(self._analysis_finished)
        self.inference_thread.start()

    def _analysis_succeeded(self, prediction):
        self.prediction = prediction
        self.current_candidate_index = 0
        self.show_full_candidate_result = False
        self.original_panel.set_bgr_image(
            resize_radiograph_for_display(prediction["original_image"], 1200)
        )
        self._populate_results(prediction["detections"])
        self._show_candidate(0)
        self.save_button.setEnabled(True)
        self.status_label.setText(
            f"Analysis complete: {prediction['detection_count']} independent fused candidates; "
            f"{prediction['suppressed_duplicate_count']} duplicate candidates merged or suppressed."
        )

    def _analysis_failed(self, message):
        self.prediction = None
        self.status_label.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis Failed", message)

    def _analysis_finished(self):
        self._set_busy(False)
        if self.inference_thread is not None:
            self.inference_thread.deleteLater()
            self.inference_thread = None

    def _populate_results(self, detections):
        self.result_table.setRowCount(len(detections))
        level_counts = {"high": 0, "medium": 0, "low": 0, "trace": 0}
        for row, detection in enumerate(detections):
            level_counts[detection["confidence_level"]] += 1
            x1, y1, x2, y2 = [round(value) for value in detection["box"]]
            values = (
                f"#{detection['number']}",
                f"{detection['evidence_score_percent']:.2f}%",
                f"{detection['raw_fusion_confidence_percent']:.2f}%",
                self._confidence_level_label(detection),
                f"{detection['view_support']}/3",
                f"({x1}, {y1}) - ({x2}, {y2})",
            )
            foreground = QColor(detection["color_hex"])
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setForeground(foreground)
                item.setTextAlignment(Qt.AlignCenter)
                if column in (0, 1):
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                self.result_table.setItem(row, column, item)

        if not detections:
            self.summary_label.setText("No displayable candidates")
        else:
            self.summary_label.setText(
                f"Total {len(detections)} | High {level_counts['high']} | "
                f"Medium {level_counts['medium']} | Low {level_counts['low']} | "
                f"Trace {level_counts['trace']}"
            )
        self.result_table.resizeRowsToContents()

    def _show_candidate(self, index):
        if self.prediction is None:
            return
        detections = self.prediction["detections"]
        if not detections:
            self.current_candidate_index = 0
            self.result_panel.set_bgr_image(self.prediction["annotated_images"][0])
            self.page_label.setText("No candidates")
            self.current_detail_label.setText(
                "No displayable candidate was found; this does not medically rule out a fracture"
            )
            self.previous_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.view_mode_button.setEnabled(False)
            return

        index = max(0, min(int(index), len(detections) - 1))
        if index != self.current_candidate_index:
            self.show_full_candidate_result = False
        self.current_candidate_index = index
        detection = detections[index]
        self._render_current_candidate()
        self.result_table.selectRow(index)
        self.page_label.setText(f"Candidate {index + 1} / {len(detections)}")
        absorbed = detection.get("absorbed_duplicates", 0)
        absorbed_text = f" | {absorbed + 1} nearby results fused" if absorbed else ""
        self.current_detail_label.setText(
            f"{self._confidence_level_label(detection)} evidence "
            f"{detection['evidence_score_percent']:.2f}% | "
            f"Raw fusion {detection['raw_fusion_confidence_percent']:.2f}% | "
            f"{detection['view_support']}/3 view support{absorbed_text}"
        )
        self._refresh_navigation_buttons()

    def _render_current_candidate(self):
        if self.prediction is None or not self.prediction["detections"]:
            return
        index = self.current_candidate_index
        if self.show_full_candidate_result:
            image = self.prediction["annotated_images"][index]
            self.view_mode_button.setText("Focus Candidate")
        else:
            detection = self.prediction["detections"][index]
            image = self._build_candidate_focus_image(detection)
            self.view_mode_button.setText("Show Full Result")
        self.result_panel.set_bgr_image(image)

    def _build_candidate_focus_image(self, detection):
        """Create a contextual close-up in which a small candidate is visible."""
        image = self.prediction["original_image"].copy()
        image_height, image_width = image.shape[:2]
        color = tuple(int(value) for value in detection["color_bgr"])

        mask = detection.get("mask")
        if mask is not None:
            if mask.shape[:2] != (image_height, image_width):
                mask = cv2.resize(
                    mask.astype("uint8"),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask = mask.astype(bool)
            if mask.any():
                overlay = image.copy()
                overlay[mask] = color
                image = cv2.addWeighted(overlay, 0.38, image, 0.62, 0.0)

        x1, y1, x2, y2 = [float(value) for value in detection["box"]]
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        crop_width = min(
            image_width,
            max(box_width * 4.5, image_width * 0.30),
        )
        crop_height = min(
            image_height,
            max(box_height * 4.5, image_height * 0.30),
        )
        crop_width = min(image_width, max(crop_width, crop_height * 0.85))
        crop_height = min(image_height, max(crop_height, crop_width * 0.72))

        crop_x1 = int(round(center_x - crop_width / 2.0))
        crop_y1 = int(round(center_y - crop_height / 2.0))
        crop_x1 = max(0, min(crop_x1, image_width - int(round(crop_width))))
        crop_y1 = max(0, min(crop_y1, image_height - int(round(crop_height))))
        crop_x2 = min(image_width, crop_x1 + int(round(crop_width)))
        crop_y2 = min(image_height, crop_y1 + int(round(crop_height)))

        line_width = max(2, int(round(max(crop_width, crop_height) / 450.0)))
        cv2.rectangle(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            line_width,
            cv2.LINE_AA,
        )
        return image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    def _toggle_candidate_view(self):
        if self.prediction is None or not self.prediction["detections"]:
            return
        self.show_full_candidate_result = not self.show_full_candidate_result
        self._render_current_candidate()

    @classmethod
    def _confidence_level_label(cls, detection):
        """Return an English confidence-level label across inference schemas."""
        explicit = detection.get("confidence_level_label")
        if explicit:
            return explicit
        return cls.LEVEL_LABELS.get(
            detection.get("confidence_level", "low"),
            "Low",
        )

    def _refresh_navigation_buttons(self):
        count = self.prediction["detection_count"] if self.prediction else 0
        self.previous_button.setEnabled(count > 0 and self.current_candidate_index > 0)
        self.next_button.setEnabled(count > 0 and self.current_candidate_index < count - 1)
        self.view_mode_button.setEnabled(count > 0)

    def _show_previous_candidate(self):
        self._show_candidate(self.current_candidate_index - 1)

    def _show_next_candidate(self):
        self._show_candidate(self.current_candidate_index + 1)

    def _jump_to_candidate(self, row, _column):
        self._show_candidate(row)

    @staticmethod
    def _available_result_directory(parent, stem):
        candidate = parent / f"{stem}_fracture_analysis"
        suffix = 2
        while candidate.exists():
            candidate = parent / f"{stem}_fracture_analysis_{suffix:02d}"
            suffix += 1
        return candidate

    def _save_result(self):
        if self.prediction is None:
            return
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Select a Parent Folder for the Result Set",
            str(self.image_path.parent if self.image_path else ROOT),
        )
        if not selected_directory:
            return

        output_directory = self._available_result_directory(
            Path(selected_directory).expanduser().resolve(),
            self.image_path.stem if self.image_path else "xray",
        )
        try:
            output_directory.mkdir(parents=False, exist_ok=False)
            original_path = output_directory / "original.png"
            save_png(
                original_path,
                resize_radiograph_for_display(self.prediction["original_image"], 1200),
            )

            result_paths = []
            if self.prediction["detections"]:
                for index, result_image in enumerate(self.prediction["annotated_images"], start=1):
                    result_path = output_directory / f"candidate_{index:02d}.png"
                    save_png(result_path, result_image)
                    result_paths.append(result_path)
            else:
                result_path = output_directory / "no_candidate.png"
                save_png(result_path, self.prediction["annotated_images"][0])
                result_paths.append(result_path)

            serialized_detections = []
            for index, detection in enumerate(self.prediction["detections"]):
                record = serializable_detection(detection)
                record["result_image"] = result_paths[index].name
                serialized_detections.append(record)

            report = {
                "model": self.prediction["model_path"],
                "device": self.prediction["device"],
                "method": self.prediction["method"],
                "evidence_normalization": self.prediction["evidence_normalization"],
                "detection_count": self.prediction["detection_count"],
                "raw_fused_candidate_count": self.prediction["raw_fused_candidate_count"],
                "suppressed_duplicate_count": self.prediction["suppressed_duplicate_count"],
                "source_image": str(self.image_path) if self.image_path else None,
                "output_directory": str(output_directory),
                "original_image": original_path.name,
                "detections": serialized_detections,
                "result_images": [path.name for path in result_paths],
                "confidence_note": (
                    "Evidence Score is validation-threshold normalized and is not a medical "
                    "probability. Raw Fusion is preserved for transparency."
                ),
                "result_json": "result.json",
            }
            (output_directory / "result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as error:
            QMessageBox.critical(self, "Save Failed", str(error))
            return

        self.status_label.setText(f"Complete result set saved: {output_directory}")
        self._show_save_toast(output_directory)

    def _show_save_toast(self, output_directory):
        self.last_saved_directory = Path(output_directory).resolve()
        self.toast_message.setText(
            f"{self.last_saved_directory.name}\n"
            "Original image, candidate images, and JSON are ready."
        )
        self._position_save_toast()
        self.save_toast.show()
        self.save_toast.raise_()
        self.toast_timer.start(7000)

    def _hide_save_toast(self):
        self.toast_timer.stop()
        self.save_toast.hide()

    def _open_saved_directory(self):
        if self.last_saved_directory is None:
            return
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.last_saved_directory))
        )
        self._hide_save_toast()

    def _position_save_toast(self):
        if not hasattr(self, "save_toast"):
            return
        parent = self.centralWidget()
        margin = 24
        x_position = max(
            margin,
            parent.width() - self.save_toast.width() - margin,
        )
        y_position = max(
            margin,
            parent.height() - self.save_toast.height() - margin,
        )
        self.save_toast.move(x_position, y_position)

    def _clear(self):
        self._hide_save_toast()
        self.image_path = None
        self.original_image = None
        self.prediction = None
        self.current_candidate_index = 0
        self.show_full_candidate_result = False
        self.original_panel.clear_image()
        self.result_panel.clear_image()
        self.result_table.setRowCount(0)
        self.summary_label.setText("Not analyzed")
        self.page_label.setText("Not analyzed")
        self.current_detail_label.setText(
            "Each independent candidate is displayed separately to prevent overlap"
        )
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.view_mode_button.setText("Show Full Result")
        self.view_mode_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.status_label.setText("Cleared. Select another image.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "save_toast") and self.save_toast.isVisible():
            self._position_save_toast()

    def closeEvent(self, event):
        if self.inference_thread is not None and self.inference_thread.isRunning():
            QMessageBox.information(
                self,
                "Analysis in Progress",
                "Wait for the current analysis to finish before closing the window.",
            )
            event.ignore()
            return
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Fracture Detection and Localization System")
    window = FractureDetectionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
