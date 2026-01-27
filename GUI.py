import os
import sys
import json
import subprocess
import re
import shutil
import urllib.request
from urllib.parse import urlparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel,
    QListWidget, QListWidgetItem, QTextBrowser, QFileDialog, QFormLayout, QGridLayout, QDoubleSpinBox,
    QSpinBox, QDialog, QGroupBox, QHBoxLayout, QCheckBox, QSizePolicy, QSplitter, QMessageBox,
    QTextEdit, QTabWidget, QComboBox, QProgressBar, QInputDialog, QStyle, QToolButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent, QTimer
from ultralytics import YOLO
from Core.semiauto_dataset_collector import ScreenCapture
from Core.train import train_yolo

# -------------------- Training Thread --------------------
class TrainerThread(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, config, params):
        super().__init__()
        self.config = config
        self.params = params
        self._stop_flag = False

    def run(self):
        def log(msg):
            self.log_signal.emit(msg)

        try:
            exit_code = train_yolo(
                model_path=self.config.get("model_path", "models/yolov12s.pt"),
                data_yaml=self.params["data_yaml"],
                epochs=self.params["epochs"],
                imgsz=self.params["imgsz"],
                batch=self.params["batch"],
                device=None,
                project=self.params["project"],
                name="auto",
                resume=self.params["resume"],
                exist_ok=self.params["exist_ok"],
                save_period=self.params["save_period"],
                patience=self.params["patience"],
                overrides=self.params.get("overrides"),
                log=log
            )
            self.finished.emit(exit_code == 0)
        except Exception as e:
            log(f"[ERROR] Exception during training: {e}")
            self.finished.emit(False)

    def stop(self):
        self._stop_flag = True


# -------------------- Capture Thread --------------------
class CaptureThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(self, capture_instance):
        super().__init__()
        self.capture_instance = capture_instance
        self.capture_instance.on_frame_ready = self.emit_frame
        self.capture_instance.log = lambda msg: self.log_signal.emit(str(msg))

    def emit_frame(self, frame):
        self.frame_signal.emit(frame)

    def run(self):
        self.capture_instance.capture_and_display()

# -------------------- StreamCut Thread --------------------
class StreamCutThread(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(bool)
    download_info_signal = pyqtSignal(dict)
    download_progress_signal = pyqtSignal(dict)

    def __init__(self, config_path, mode="all"):
        super().__init__()
        self.config_path = config_path
        self.mode = mode
        self.stop_requested = False
        self.stream_cut = None

    def run(self):
        try:
            from Core.StreamCut import StreamCut
            stream_cut = StreamCut(self.config_path)
            self.stream_cut = stream_cut
            stream_cut.log = lambda msg: self.log_signal.emit(str(msg))
            stream_cut.on_download_info = lambda info: self.download_info_signal.emit(info)
            stream_cut.on_download_progress = lambda info: self.download_progress_signal.emit(info)
            if self.mode == "download_only":
                stream_cut.download_all()
            elif self.mode == "process_only":
                stream_cut.split_all()
                stream_cut.process_all()
                stream_cut.log("[DONE]")
            else:
                stream_cut.run()
            self.finished.emit(True)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] {e}")
            self.finished.emit(False)

    def request_stop(self):
        self.stop_requested = True
        if self.stream_cut:
            self.stream_cut.stop()


# -------------------- Benchmark Thread --------------------
class BenchmarkThread(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, models_dir=None, images_dir=None):
        super().__init__()
        self.models_dir = models_dir
        self.images_dir = images_dir

    def run(self):
        try:
            script_path = Path(__file__).resolve().parent / "benchmark" / "benchmark.py"
            cmd = [sys.executable, str(script_path)]
            if self.models_dir:
                cmd += ["--models", self.models_dir]
            if self.images_dir:
                cmd += ["--images-dir", self.images_dir]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in proc.stdout:
                self.log_signal.emit(line.rstrip())
            proc.wait()
            self.finished.emit(proc.returncode == 0)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Benchmark failed: {e}")
            self.finished.emit(False)

# -------------------- Model Download Thread --------------------
class ModelDownloadThread(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, model_ref, dest_folder):
        super().__init__()
        self.model_ref = model_ref.strip()
        self.dest_folder = dest_folder

    def run(self):
        if not self.model_ref:
            self.log_signal.emit("[ERROR] Model name or URL is empty.")
            self.finished.emit(False, "")
            return

        try:
            Path(self.dest_folder).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Failed to create folder: {e}")
            self.finished.emit(False, "")
            return

        if self.model_ref.startswith(("http://", "https://")):
            try:
                name = Path(urlparse(self.model_ref).path).name or "model.pt"
                dest_path = Path(self.dest_folder) / name
                self.log_signal.emit(f"[INFO] Downloading: {self.model_ref}")
                urllib.request.urlretrieve(self.model_ref, dest_path)
                self.log_signal.emit(f"[INFO] Saved: {dest_path}")
                self.finished.emit(True, str(dest_path))
            except Exception as e:
                self.log_signal.emit(f"[ERROR] Download failed: {e}")
                self.finished.emit(False, "")
            return

        name = self.model_ref
        if not Path(name).suffix:
            name = f"{name}.pt"
        dest_path = Path(self.dest_folder) / name
        if dest_path.exists():
            self.log_signal.emit(f"[INFO] Already exists: {dest_path}")
            self.finished.emit(True, str(dest_path))
            return

        try:
            from ultralytics import YOLO
        except Exception as e:
            self.log_signal.emit(f"[ERROR] Ultralytics not available: {e}")
            self.finished.emit(False, "")
            return

        try:
            self.log_signal.emit(f"[INFO] Fetching model: {name}")
            YOLO(name)
        except Exception as e:
            if "yolo26" in name:
                self.log_signal.emit("[ERROR] YOLO26 weights may be unavailable. Try a direct URL.")
            self.log_signal.emit(f"[ERROR] Download failed: {e}")
            self.finished.emit(False, "")
            return

        if dest_path.exists():
            self.log_signal.emit(f"[INFO] Saved: {dest_path}")
            self.finished.emit(True, str(dest_path))
            return

        # Some ultralytics versions save into CWD; move it into the requested folder.
        cwd_candidate = Path.cwd() / name
        script_candidate = Path(__file__).resolve().parent / name
        for candidate in (cwd_candidate, script_candidate):
            if candidate.exists():
                try:
                    shutil.move(str(candidate), str(dest_path))
                    self.log_signal.emit(f"[INFO] Moved from {candidate} to {dest_path}")
                    self.finished.emit(True, str(dest_path))
                    return
                except Exception:
                    pass

        cache_roots = [
            Path.home() / ".cache" / "ultralytics",
            Path.home() / ".cache" / "torch" / "hub",
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        ]
        for root in cache_roots:
            if not root.exists():
                continue
            try:
                hit = next(root.rglob(name))
                shutil.copy2(hit, dest_path)
                self.log_signal.emit(f"[INFO] Copied from cache: {dest_path}")
                self.finished.emit(True, str(dest_path))
                return
            except StopIteration:
                continue
            except Exception:
                continue

        self.log_signal.emit("[WARNING] Model downloaded to cache, but local file not found.")
        self.finished.emit(False, "")

# -------------------- StreamCut Dialog --------------------
class StreamCutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("StreamCut")
        self.setGeometry(150, 150, 700, 700)
        self.config = {}
        self.stream_thread = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_config)
        self._stream_mode = "all"
        self.init_ui()
        self.load_config(self.config_path_input.text())

    def create_download_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Download progress")
        dialog.setModal(False)
        dialog.setMinimumWidth(360)
        layout = QFormLayout()

        dialog.title_label = QLabel("-")
        dialog.size_label = QLabel("-")
        dialog.downloaded_label = QLabel("-")
        dialog.speed_label = QLabel("-")
        dialog.eta_label = QLabel("-")
        dialog.progress_bar = QProgressBar()
        dialog.progress_bar.setRange(0, 100)

        layout.addRow("Title:", dialog.title_label)
        layout.addRow("Size:", dialog.size_label)
        layout.addRow("Downloaded:", dialog.downloaded_label)
        layout.addRow("Speed:", dialog.speed_label)
        layout.addRow("ETA:", dialog.eta_label)
        layout.addRow("Progress:", dialog.progress_bar)
        dialog.setLayout(layout)
        return dialog

    def init_ui(self):
        layout = QVBoxLayout()

        path_layout = QHBoxLayout()
        default_path = Path(__file__).resolve().parent / "configs" / "configStreamCut.json"
        self.config_path_input = QLineEdit(str(default_path))
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_config_path)
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(lambda: self.load_config(self.config_path_input.text()))
        path_layout.addWidget(QLabel("Config:"))
        path_layout.addWidget(self.config_path_input)
        path_layout.addWidget(btn_browse)
        path_layout.addWidget(btn_load)
        layout.addLayout(path_layout)

        form = QFormLayout()

        self.video_sources_list = QListWidget()
        self.video_sources_list.setToolTip("List of VOD URLs. Check to download. Downloaded items show an icon.")
        self.video_sources_list.setSelectionMode(QListWidget.ExtendedSelection)
        form.addRow("Video sources:", self.video_sources_list)

        sources_btns = QHBoxLayout()
        btn_add_source = QPushButton("Add")
        btn_add_source.setToolTip("Add a new source line.")
        btn_add_source.clicked.connect(self.add_video_source)
        btn_sync_sources = QPushButton("Sync")
        btn_sync_sources.setToolTip("Mark already downloaded streams.")
        btn_sync_sources.clicked.connect(self.sync_downloaded_sources)
        btn_remove_sources = QPushButton("Remove")
        btn_remove_sources.setToolTip("Remove selected sources.")
        btn_remove_sources.clicked.connect(self.remove_selected_sources)
        btn_check_all = QPushButton("Check all")
        btn_check_all.clicked.connect(lambda: self.set_sources_check_state(Qt.Checked))
        btn_uncheck_all = QPushButton("Uncheck all")
        btn_uncheck_all.clicked.connect(lambda: self.set_sources_check_state(Qt.Unchecked))
        sources_btns.addWidget(btn_add_source)
        sources_btns.addWidget(btn_sync_sources)
        sources_btns.addWidget(btn_remove_sources)
        sources_btns.addWidget(btn_check_all)
        sources_btns.addWidget(btn_uncheck_all)
        form.addRow("", sources_btns)

        self.use_selected_only_checkbox = QCheckBox("Download only checked sources")
        self.use_selected_only_checkbox.setToolTip("If enabled, StreamCut will download only checked sources.")
        self.use_selected_only_checkbox.setChecked(True)
        form.addRow("", self.use_selected_only_checkbox)

        self.video_sources_list.itemChanged.connect(lambda _: self.schedule_save_config())
        self.use_selected_only_checkbox.toggled.connect(lambda _: self.schedule_save_config())

        self.classes_input = QTextEdit()
        self.classes_input.setPlaceholderText("One class per line")
        self.classes_input.setToolTip("Class filter: keep only these classes.")
        form.addRow("Classes:", self.classes_input)

        self.raw_stream_folder_input = QLineEdit()
        self.raw_stream_folder_input.setToolTip("Folder for downloaded VODs.")
        raw_layout = QHBoxLayout()
        raw_layout.addWidget(self.raw_stream_folder_input)
        btn_raw = QPushButton("...")
        btn_raw.setFixedWidth(32)
        btn_raw.clicked.connect(lambda: self.browse_folder(self.raw_stream_folder_input))
        raw_layout.addWidget(btn_raw)
        form.addRow("Raw stream folder:", raw_layout)

        self.chunks_folder_input = QLineEdit()
        self.chunks_folder_input.setToolTip("Folder for .ts segments.")
        chunks_layout = QHBoxLayout()
        chunks_layout.addWidget(self.chunks_folder_input)
        btn_chunks = QPushButton("...")
        btn_chunks.setFixedWidth(32)
        btn_chunks.clicked.connect(lambda: self.browse_folder(self.chunks_folder_input))
        chunks_layout.addWidget(btn_chunks)
        form.addRow("Chunks folder:", chunks_layout)

        self.output_folder_input = QLineEdit()
        self.output_folder_input.setToolTip("Dataset output folder (images/labels).")
        out_layout = QHBoxLayout()
        out_layout.addWidget(self.output_folder_input)
        btn_out = QPushButton("...")
        btn_out.setFixedWidth(32)
        btn_out.clicked.connect(lambda: self.browse_folder(self.output_folder_input))
        out_layout.addWidget(btn_out)
        form.addRow("Output folder:", out_layout)

        self.model_path_input = QLineEdit()
        self.model_path_input.setToolTip("Path to YOLO .pt model for inference.")
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        btn_model = QPushButton("...")
        btn_model.setFixedWidth(32)
        btn_model.clicked.connect(lambda: self.browse_file(self.model_path_input, "Select Model File", "Model Files (*.pt)"))
        model_layout.addWidget(btn_model)
        form.addRow("Model path:", model_layout)

        self.ffmpeg_path_input = QLineEdit()
        self.ffmpeg_path_input.setToolTip("Path to ffmpeg.exe for slicing.")
        ffmpeg_layout = QHBoxLayout()
        ffmpeg_layout.addWidget(self.ffmpeg_path_input)
        btn_ffmpeg = QPushButton("...")
        btn_ffmpeg.setFixedWidth(32)
        btn_ffmpeg.clicked.connect(lambda: self.browse_file(self.ffmpeg_path_input, "Select ffmpeg.exe", "Executables (*.exe);;All Files (*.*)"))
        ffmpeg_layout.addWidget(btn_ffmpeg)
        form.addRow("FFmpeg path:", ffmpeg_layout)

        self.cookies_path_input = QLineEdit()
        self.cookies_path_input.setToolTip("Path to Twitch cookies (if needed).")
        cookies_layout = QHBoxLayout()
        cookies_layout.addWidget(self.cookies_path_input)
        btn_cookies = QPushButton("...")
        btn_cookies.setFixedWidth(32)
        btn_cookies.clicked.connect(lambda: self.browse_file(self.cookies_path_input, "Select Cookies File", "All Files (*.*)"))
        cookies_layout.addWidget(btn_cookies)
        form.addRow("Cookies path:", cookies_layout)

        self.download_archive_input = QLineEdit()
        self.download_archive_input.setToolTip("Archive file of downloaded URLs (downloaded.txt).")
        archive_layout = QHBoxLayout()
        archive_layout.addWidget(self.download_archive_input)
        btn_archive = QPushButton("...")
        btn_archive.setFixedWidth(32)
        btn_archive.clicked.connect(lambda: self.browse_save_file(self.download_archive_input, "Select download archive", "Text Files (*.txt);;All Files (*.*)"))
        archive_layout.addWidget(btn_archive)
        form.addRow("Download archive:", archive_layout)

        self.resume_info_input = QLineEdit()
        self.resume_info_input.setToolTip("Resume file (resume.json).")
        resume_layout = QHBoxLayout()
        resume_layout.addWidget(self.resume_info_input)
        btn_resume = QPushButton("...")
        btn_resume.setFixedWidth(32)
        btn_resume.clicked.connect(lambda: self.browse_save_file(self.resume_info_input, "Select resume file", "JSON Files (*.json);;All Files (*.*)"))
        resume_layout.addWidget(btn_resume)
        form.addRow("Resume info file:", resume_layout)

        self.pause_after_download_checkbox = QCheckBox("Pause after download")
        self.pause_after_download_checkbox.setToolTip("Ask before processing downloaded streams.")
        self.pause_after_download_checkbox.setChecked(True)
        form.addRow("", self.pause_after_download_checkbox)
        self.pause_after_download_checkbox.toggled.connect(lambda _: self.schedule_save_config())

        self.time_interval_input = QSpinBox()
        self.time_interval_input.setRange(1, 100000)
        self.time_interval_input.setToolTip("Run inference every Nth frame.")

        self.detection_threshold_input = QDoubleSpinBox()
        self.detection_threshold_input.setRange(0.0, 1.0)
        self.detection_threshold_input.setSingleStep(0.05)
        self.detection_threshold_input.setToolTip("Confidence threshold for saving detections.")

        self.chunks_per_stream_input = QSpinBox()
        self.chunks_per_stream_input.setRange(1, 10000)
        self.chunks_per_stream_input.setToolTip("How many parts to split each VOD into.")

        self.max_download_workers_input = QSpinBox()
        self.max_download_workers_input.setRange(1, 128)
        self.max_download_workers_input.setToolTip("Parallel download workers (yt-dlp).")

        self.split_workers_input = QSpinBox()
        self.split_workers_input.setRange(1, 128)
        self.split_workers_input.setToolTip("Workers for slicing .ts (ffmpeg).")

        self.process_workers_input = QSpinBox()
        self.process_workers_input.setRange(1, 128)
        self.process_workers_input.setToolTip("Workers for YOLO inference per chunk.")

        self.save_workers_input = QSpinBox()
        self.save_workers_input.setRange(1, 128)
        self.save_workers_input.setToolTip("Workers for saving images/labels to disk.")
        form.addRow(QLabel("Processing & Workers"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)

        lbl_time = QLabel("Time interval:")
        lbl_time.setToolTip(self.time_interval_input.toolTip())
        grid.addWidget(lbl_time, 0, 0)
        grid.addWidget(self.time_interval_input, 0, 1)

        lbl_det = QLabel("Detection threshold:")
        lbl_det.setToolTip(self.detection_threshold_input.toolTip())
        grid.addWidget(lbl_det, 0, 2)
        grid.addWidget(self.detection_threshold_input, 0, 3)

        lbl_chunks = QLabel("Chunks per stream:")
        lbl_chunks.setToolTip(self.chunks_per_stream_input.toolTip())
        grid.addWidget(lbl_chunks, 1, 0)
        grid.addWidget(self.chunks_per_stream_input, 1, 1)

        lbl_dl = QLabel("Download workers:")
        lbl_dl.setToolTip(self.max_download_workers_input.toolTip())
        grid.addWidget(lbl_dl, 1, 2)
        grid.addWidget(self.max_download_workers_input, 1, 3)

        lbl_split = QLabel("Split workers:")
        lbl_split.setToolTip(self.split_workers_input.toolTip())
        grid.addWidget(lbl_split, 2, 0)
        grid.addWidget(self.split_workers_input, 2, 1)

        lbl_proc = QLabel("Process workers:")
        lbl_proc.setToolTip(self.process_workers_input.toolTip())
        grid.addWidget(lbl_proc, 2, 2)
        grid.addWidget(self.process_workers_input, 2, 3)

        lbl_save = QLabel("Save workers:")
        lbl_save.setToolTip(self.save_workers_input.toolTip())
        grid.addWidget(lbl_save, 3, 0)
        grid.addWidget(self.save_workers_input, 3, 1)

        grid_widget = QWidget()
        grid_widget.setLayout(grid)
        form.addRow(grid_widget)

        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save Config")
        self.btn_save.clicked.connect(self.save_config)
        self.btn_run = QPushButton("Run StreamCut")
        self.btn_run.clicked.connect(self.run_streamcut)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_streamcut)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.console = QTextBrowser()
        self.console.setMinimumHeight(150)
        layout.addWidget(self.console)

        self._icon_downloaded = self.style().standardIcon(QStyle.SP_DialogApplyButton)

        self.setLayout(layout)

    def log_to_console(self, message):
        self.console.append(message)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def schedule_save_config(self):
        if self._save_timer:
            self._save_timer.start(400)

    def add_video_source(self):
        text, ok = QInputDialog.getText(self, "Add source", "Video URL:")
        if not ok:
            return
        self.add_video_sources_from_text(text)

    def remove_selected_sources(self):
        for item in self.video_sources_list.selectedItems():
            row = self.video_sources_list.row(item)
            self.video_sources_list.takeItem(row)
        self.schedule_save_config()

    def add_video_sources_from_text(self, text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return
        existing = {self.video_sources_list.item(i).text() for i in range(self.video_sources_list.count())}
        for line in lines:
            if line in existing:
                continue
            item = QListWidgetItem(line)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(Qt.Checked)
            self.video_sources_list.addItem(item)
        self.schedule_save_config()

    def browse_config_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select StreamCut config", "", "JSON Files (*.json)")
        if path:
            self.config_path_input.setText(path)
            self.load_config(path)

    def browse_file(self, target_input, title, filter_str):
        start = target_input.text().strip()
        path, _ = QFileDialog.getOpenFileName(self, title, start, filter_str)
        if path:
            target_input.setText(path)

    def browse_save_file(self, target_input, title, filter_str):
        start = target_input.text().strip()
        path, _ = QFileDialog.getSaveFileName(self, title, start, filter_str)
        if path:
            target_input.setText(path)

    def browse_folder(self, target_input):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", target_input.text().strip())
        if folder:
            target_input.setText(folder)


    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.config = {}
            self.log_to_console(f"[ERROR] Failed to load config: {e}")

        self.classes_input.setPlainText("\n".join(self.config.get("classes", [])))
        self.use_selected_only_checkbox.setChecked(bool(self.config.get("use_selected_only", True)))
        self.pause_after_download_checkbox.setChecked(bool(self.config.get("pause_after_download", True)))
        self.video_sources_list.clear()
        sources = self.config.get("video_sources", [])
        selected = set(self.config.get("selected_video_sources", []))
        for source in sources:
            item = QListWidgetItem(source)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            if selected:
                item.setCheckState(Qt.Checked if source in selected else Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)
            self.video_sources_list.addItem(item)
        self.raw_stream_folder_input.setText(self.config.get("raw_stream_folder", ""))
        self.chunks_folder_input.setText(self.config.get("chunks_folder", ""))
        self.output_folder_input.setText(self.config.get("output_folder", ""))
        self.model_path_input.setText(self.config.get("model_path", ""))
        self.ffmpeg_path_input.setText(self.config.get("ffmpeg_path", ""))
        self.cookies_path_input.setText(self.config.get("twitch_cookies_path", ""))
        self.download_archive_input.setText(self.config.get("download_archive", ""))
        self.resume_info_input.setText(self.config.get("resume_info_file", ""))
        self.time_interval_input.setValue(int(self.config.get("time_interval", 1)))
        self.detection_threshold_input.setValue(float(self.config.get("detection_threshold", 0.5)))
        self.chunks_per_stream_input.setValue(int(self.config.get("chunks_per_stream", 1)))
        self.max_download_workers_input.setValue(int(self.config.get("max_download_workers", 1)))
        self.split_workers_input.setValue(int(self.config.get("split_workers", 1)))
        self.process_workers_input.setValue(int(self.config.get("process_workers", 1)))
        self.save_workers_input.setValue(int(self.config.get("save_workers", 1)))
        self.sync_downloaded_sources()

    def update_config_from_fields(self):
        all_sources = [self.video_sources_list.item(i).text() for i in range(self.video_sources_list.count())]
        selected_sources = self.get_checked_sources()
        self.config["video_sources"] = all_sources
        self.config["selected_video_sources"] = selected_sources
        self.config["use_selected_only"] = bool(self.use_selected_only_checkbox.isChecked())
        self.config["pause_after_download"] = bool(self.pause_after_download_checkbox.isChecked())
        self.config["classes"] = [line.strip() for line in self.classes_input.toPlainText().splitlines() if line.strip()]
        self.config["raw_stream_folder"] = self.raw_stream_folder_input.text().strip()
        self.config["chunks_folder"] = self.chunks_folder_input.text().strip()
        self.config["output_folder"] = self.output_folder_input.text().strip()
        self.config["model_path"] = self.model_path_input.text().strip()
        self.config["ffmpeg_path"] = self.ffmpeg_path_input.text().strip()
        self.config["twitch_cookies_path"] = self.cookies_path_input.text().strip()
        self.config["download_archive"] = self.download_archive_input.text().strip()
        self.config["resume_info_file"] = self.resume_info_input.text().strip()
        self.config["time_interval"] = int(self.time_interval_input.value())
        self.config["detection_threshold"] = float(self.detection_threshold_input.value())
        self.config["chunks_per_stream"] = int(self.chunks_per_stream_input.value())
        self.config["max_download_workers"] = int(self.max_download_workers_input.value())
        self.config["split_workers"] = int(self.split_workers_input.value())
        self.config["process_workers"] = int(self.process_workers_input.value())
        self.config["save_workers"] = int(self.save_workers_input.value())

    def set_sources_check_state(self, state):
        for i in range(self.video_sources_list.count()):
            self.video_sources_list.item(i).setCheckState(state)

    def get_checked_sources(self):
        sources = []
        for i in range(self.video_sources_list.count()):
            item = self.video_sources_list.item(i)
            if item.checkState() == Qt.Checked:
                sources.append(item.text())
        return sources

    def _extract_id_from_url(self, url):
        url = url.strip()
        if not url:
            return None
        patterns = [
            r"twitch\.tv/videos/(?P<id>\d+)",
            r"youtu\.be/(?P<id>[\w-]+)",
            r"youtube\.com/watch\?v=(?P<id>[\w-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group("id")
        if re.fullmatch(r"[\w-]+", url):
            return url
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "ignoreconfig": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get("id")
        except Exception:
            return None

    def sync_downloaded_sources(self):
        raw_folder = self.raw_stream_folder_input.text().strip()
        archive_path = self.download_archive_input.text().strip()
        downloaded_ids = set()
        if archive_path and os.path.isfile(archive_path):
            try:
                with open(archive_path, "r", encoding="utf-8", errors="ignore") as f:
                    downloaded_ids = {line.strip() for line in f if line.strip()}
            except Exception as e:
                self.log_to_console(f"[ERROR] Failed to read archive: {e}")
        downloaded_files = set()
        if raw_folder and os.path.isdir(raw_folder):
            try:
                downloaded_files = {p.stem for p in Path(raw_folder).glob("*.*")}
            except Exception:
                downloaded_files = set()

        if not downloaded_ids and not downloaded_files:
            self.log_to_console("[INFO] No downloaded archive/files found.")
            return

        marked = 0
        for i in range(self.video_sources_list.count()):
            item = self.video_sources_list.item(i)
            url = item.text()
            vid = self._extract_id_from_url(url)
            if not vid:
                item.setIcon(QIcon())
                continue
            if vid in downloaded_files or any(line.endswith(f":{vid}") or line == vid for line in downloaded_ids):
                item.setIcon(self._icon_downloaded)
                marked += 1
            else:
                item.setIcon(QIcon())

        self.log_to_console(f"[INFO] Sync complete. Marked {marked} downloaded sources.")

    def save_config(self):
        path = self.config_path_input.text().strip()
        if not path:
            self.log_to_console("[ERROR] Config path is empty.")
            return

        self.update_config_from_fields()
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.log_to_console("[INFO] Config saved.")
        except Exception as e:
            self.log_to_console(f"[ERROR] Failed to save config: {e}")

    def run_streamcut(self):
        if self.stream_thread and self.stream_thread.isRunning():
            self.log_to_console("[INFO] StreamCut is already running.")
            return

        self.save_config()
        config_path = self.config_path_input.text().strip()
        if not config_path:
            return

        self._download_start_ts = time.time()
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running...")
        self.btn_stop.setEnabled(True)
        pause_after = bool(self.pause_after_download_checkbox.isChecked())
        self._stream_mode = "download_only" if pause_after else "all"
        self.stream_thread = StreamCutThread(config_path, mode=self._stream_mode)
        self.stream_thread.log_signal.connect(self.log_to_console)
        self.stream_thread.download_info_signal.connect(self.on_download_info)
        self.stream_thread.download_progress_signal.connect(self.on_download_progress)
        self.stream_thread.finished.connect(self.on_streamcut_finished)
        self.stream_thread.start()

        if self._stream_mode in ("download_only", "all"):
            self.download_dialog = self.create_download_dialog()
            self.download_dialog.show()

    def stop_streamcut(self):
        if self.stream_thread and self.stream_thread.isRunning():
            self.log_to_console("[INFO] Stop requested. Waiting for current task...")
            self.stream_thread.request_stop()
            self.btn_stop.setEnabled(False)
        else:
            self.btn_stop.setEnabled(False)

    def on_streamcut_finished(self, success):
        if not success:
            self.log_to_console("[ERROR] StreamCut stopped with errors.")
            self.btn_run.setEnabled(True)
            self.btn_run.setText("Run StreamCut")
            self.btn_stop.setEnabled(False)
            if hasattr(self, "download_dialog"):
                self.download_dialog.close()
            return

        if self._stream_mode == "download_only":
            self.log_to_console("[INFO] Download complete. Waiting for confirmation…")
            self.show_download_sizes()
            self.btn_stop.setEnabled(False)
            reply = QMessageBox.question(
                self,
                "Continue processing?",
                "Downloads finished. Continue with splitting and inference?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.log_to_console("[INFO] продолжaем обработку…")
                self._stream_mode = "process_only"
                config_path = self.config_path_input.text().strip()
                self.stream_thread = StreamCutThread(config_path, mode="process_only")
                self.stream_thread.log_signal.connect(self.log_to_console)
                self.stream_thread.finished.connect(self.on_streamcut_finished)
                self.btn_stop.setEnabled(True)
                self.stream_thread.start()
                return
            else:
                self.log_to_console("[INFO] Processing cancelled by user.")
        else:
            self.log_to_console("[INFO] StreamCut finished.")

        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run StreamCut")
        self.btn_stop.setEnabled(False)
        if hasattr(self, "download_dialog"):
            self.download_dialog.close()

    def on_download_info(self, info):
        if not hasattr(self, "download_dialog"):
            return
        title = info.get("title", "-")
        size = self.format_size(info.get("size_bytes"))
        self.download_dialog.title_label.setText(title)
        self.download_dialog.size_label.setText(size)

    def on_download_progress(self, info):
        if not hasattr(self, "download_dialog"):
            return
        downloaded = self.format_size(info.get("downloaded_bytes"))
        total = self.format_size(info.get("total_bytes"))
        speed = self.format_size(info.get("speed_bytes")) + "/s" if info.get("speed_bytes") else "-"
        eta = info.get("eta")
        if eta is None:
            eta_str = "-"
        else:
            mins, secs = divmod(int(eta), 60)
            eta_str = f"{mins:02d}:{secs:02d}"
        percent = int(info.get("percent", 0))
        self.download_dialog.downloaded_label.setText(f"{downloaded} / {total}")
        self.download_dialog.speed_label.setText(speed)
        self.download_dialog.eta_label.setText(eta_str)
        self.download_dialog.progress_bar.setValue(percent)

    def show_download_sizes(self):
        raw_folder = self.raw_stream_folder_input.text().strip()
        if not raw_folder or not os.path.isdir(raw_folder):
            return
        start_ts = getattr(self, "_download_start_ts", None)
        if start_ts is None:
            return

        files = []
        for p in Path(raw_folder).glob("*.*"):
            try:
                if p.stat().st_mtime >= start_ts - 2:
                    files.append(p)
            except Exception:
                continue

        if not files:
            return

        def fmt_size(num):
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if num < 1024.0:
                    return f"{num:.2f} {unit}"
                num /= 1024.0
            return f"{num:.2f} PB"

        lines = []
        for f in sorted(files):
            try:
                size = fmt_size(f.stat().st_size)
            except Exception:
                size = "unknown"
            lines.append(f"{f.name} — {size}")

        QMessageBox.information(
            self,
            "Downloaded files",
            "Downloaded streams:\n\n" + "\n".join(lines)
        )

    @staticmethod
    def format_size(num):
        if num is None:
            return "-"
        try:
            num = float(num)
        except Exception:
            return "-"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num < 1024.0:
                return f"{num:.2f} {unit}"
            num /= 1024.0
        return f"{num:.2f} PB"

# -------------------- Benchmark Dialog --------------------
class BenchmarkDialog(QDialog):
    def __init__(self, log_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Benchmark")
        self.setGeometry(200, 200, 600, 200)
        self.log_callback = log_callback
        self.thread = None
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.models_dir_input = QLineEdit()
        models_layout = QHBoxLayout()
        models_layout.addWidget(self.models_dir_input)
        btn_models = QPushButton("...")
        btn_models.setFixedWidth(32)
        btn_models.clicked.connect(lambda: self.browse_folder(self.models_dir_input))
        models_layout.addWidget(btn_models)
        layout.addRow("Models dir:", models_layout)

        self.images_dir_input = QLineEdit()
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.images_dir_input)
        btn_images = QPushButton("...")
        btn_images.setFixedWidth(32)
        btn_images.clicked.connect(lambda: self.browse_folder(self.images_dir_input))
        images_layout.addWidget(btn_images)
        layout.addRow("Images dir:", images_layout)

        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Benchmark")
        self.btn_run.clicked.connect(self.run_benchmark)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_run)
        layout.addRow("", btn_layout)

        self.setLayout(layout)

    def browse_folder(self, target_input):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", target_input.text().strip())
        if folder:
            target_input.setText(folder)

    def run_benchmark(self):
        if self.thread and self.thread.isRunning():
            return
        models_dir = self.models_dir_input.text().strip() or None
        images_dir = self.images_dir_input.text().strip() or None
        self.btn_run.setEnabled(False)
        self.thread = BenchmarkThread(models_dir=models_dir, images_dir=images_dir)
        if self.log_callback:
            self.thread.log_signal.connect(self.log_callback)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, ok):
        if self.log_callback:
            self.log_callback("[INFO] Benchmark finished." if ok else "[ERROR] Benchmark failed.")
        self.btn_run.setEnabled(True)

# -------------------- Config Dialog --------------------
class ConfigDialog(QDialog):
    def __init__(self, config, config_path):
        super().__init__()
        self.setWindowTitle("Configuration Settings")
        self.setGeometry(200, 200, 400, 400)
        self.config = config
        self.config_path = Path(config_path)
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        # model path
        self.model_path_input = QLineEdit(self.config.get("model_path", ""))
        self.model_path_input.setToolTip("Path to YOLO .pt model for detection.")
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        btn_model = QPushButton("...")
        btn_model.setFixedWidth(32)
        btn_model.clicked.connect(self.browse_model_path)
        model_layout.addWidget(btn_model)
        layout.addRow("Model Path:", model_layout)

        # output folder
        self.output_folder_input = QLineEdit(self.config.get("output_folder", ""))
        self.output_folder_input.setToolTip("Output folder for data collection (images/labels).")
        out_layout = QHBoxLayout()
        out_layout.addWidget(self.output_folder_input)
        btn_out = QPushButton("...")
        btn_out.setFixedWidth(32)
        btn_out.clicked.connect(self.browse_output_folder)
        out_layout.addWidget(btn_out)
        layout.addRow("Output Folder:", out_layout)

        note = QLabel("These settings affect data collection/detection only (not training).")
        note.setWordWrap(True)
        layout.addRow(note)

        # grabber / detection settings
        self.crop_size_input = QDoubleSpinBox()
        self.crop_size_input.setSingleStep(0.01)
        self.crop_size_input.setRange(0.0, 1.0)
        self.crop_size_input.setValue(self.config.get("grabber", {}).get("crop_size", 0.8))
        self.crop_size_input.setToolTip("Central crop size (0..1) for capture.")

        self.width_input = QSpinBox()
        self.width_input.setRange(1, 3840)
        self.width_input.setSingleStep(1)
        self.width_input.setValue(self.config.get("grabber", {}).get("width", 640))
        self.width_input.setToolTip("Capture width in pixels.")

        self.height_input = QSpinBox()
        self.height_input.setRange(1, 2160)
        self.height_input.setSingleStep(1)
        self.height_input.setValue(self.config.get("grabber", {}).get("height", 640))
        self.height_input.setToolTip("Capture height in pixels.")

        self.detection_threshold_input = QDoubleSpinBox()
        self.detection_threshold_input.setValue(self.config.get("detection_threshold", 0.5))
        self.detection_threshold_input.setToolTip("Confidence threshold for saving detections.")

        detect_group = QGroupBox("Detection / Capture")
        detect_grid = QGridLayout()
        detect_grid.setHorizontalSpacing(12)
        detect_grid.addWidget(QLabel("Crop size:"), 0, 0)
        detect_grid.addWidget(self.crop_size_input, 0, 1)
        detect_grid.addWidget(QLabel("Width:"), 0, 2)
        detect_grid.addWidget(self.width_input, 0, 3)
        detect_grid.addWidget(QLabel("Height:"), 1, 0)
        detect_grid.addWidget(self.height_input, 1, 1)
        detect_grid.addWidget(QLabel("Detection thr:"), 1, 2)
        detect_grid.addWidget(self.detection_threshold_input, 1, 3)
        detect_group.setLayout(detect_grid)
        layout.addRow(detect_group)

        # save interval
        self.save_interval_input = QSpinBox()
        self.save_interval_input.setValue(self.config.get("save_interval", 3))
        self.save_interval_input.setToolTip("Min interval between saves (seconds).")
        layout.addRow("Save Interval:", self.save_interval_input)

        # data folder (split/label tools)
        self.data_folder_input = QLineEdit(self.config.get("data_folder", ""))
        self.data_folder_input.setToolTip("Dataset folder with images/labels for tools (split/label). Not used for training.")
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_folder_input)
        btn_data = QPushButton("...")
        btn_data.setFixedWidth(32)
        btn_data.clicked.connect(self.browse_data_folder)
        data_layout.addWidget(btn_data)
        layout.addRow("Data Folder:", data_layout)

        # label verification folder
        self.label_data_folder_input = QLineEdit(self.config.get("label_data_folder", self.config.get("data_folder", "")))
        self.label_data_folder_input.setToolTip("Folder with images/labels used by Label Verification tool.")
        label_layout = QHBoxLayout()
        label_layout.addWidget(self.label_data_folder_input)
        btn_label = QPushButton("...")
        btn_label.setFixedWidth(32)
        btn_label.clicked.connect(self.browse_label_data_folder)
        label_layout.addWidget(btn_label)
        layout.addRow("Label Folder:", label_layout)

        # save button
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self.save_config)
        layout.addWidget(btn_save)

        self.setLayout(layout)

    def browse_model_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pt)")
        if path:
            self.model_path_input.setText(path)


    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_input.setText(folder)

    def browse_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.data_folder_input.setText(folder)

    def browse_label_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Label Folder")
        if folder:
            self.label_data_folder_input.setText(folder)

    def save_config(self):
        self.config["model_path"] = self.model_path_input.text()
        self.config["grabber"] = {
            "crop_size": self.crop_size_input.value(),
            "width": self.width_input.value(),
            "height": self.height_input.value(),
        }
        self.config["output_folder"] = self.output_folder_input.text()
        self.config["save_interval"] = self.save_interval_input.value()
        self.config["detection_threshold"] = self.detection_threshold_input.value()
        self.config["data_folder"] = self.data_folder_input.text()
        self.config["label_data_folder"] = self.label_data_folder_input.text()
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.accept()
        except Exception as e:
            print(f"Error saving config: {e}")

# -------------------- Main App --------------------
class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.btn_def = None
        self.setWindowTitle('YOLO Dataset Manager')
        self.setGeometry(100, 100, 800, 500)

        # state
        self.config_path = Path(__file__).resolve().parent / "configs" / "config.json"
        self.config = self.load_config(self.config_path)
        self.capture = None
        self.capture_thread = None

        self.training_active = False
        self.data_collection_active = False

        self.apply_ui_font(self.config.get("ui", {}).get("font_size", 11))
        self.init_ui()

    # -------- Config I/O ----------
    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def save_config(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.console.append("Configuration saved.")
        except Exception as e:
            self.console.append(f"Error saving configuration: {e}")

    # -------- UI ----------
    def init_ui(self):
        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tab_dataset = QWidget()
        self.tab_training = QWidget()
        self.tab_tools = QWidget()

        self.init_dataset_tab()
        self.init_training_tab()
        self.init_tools_tab()

        self.tabs.addTab(self.tab_dataset, "Dataset")
        self.tabs.addTab(self.tab_training, "Training")
        self.tabs.addTab(self.tab_tools, "Tools")

        font_widget = QWidget()
        font_layout = QHBoxLayout()
        font_layout.setContentsMargins(0, 0, 0, 0)
        font_layout.setSpacing(6)
        font_label = QLabel("Font")
        self.font_size_input = QSpinBox()
        self.font_size_input.setRange(8, 24)
        self.font_size_input.setValue(int(self.config.get("ui", {}).get("font_size", 11)))
        self.font_size_input.setToolTip("Adjust application font size.")
        self.font_size_input.valueChanged.connect(self.on_font_size_changed)
        font_layout.addWidget(font_label)
        font_layout.addWidget(self.font_size_input)
        font_widget.setLayout(font_layout)
        self.tabs.setCornerWidget(font_widget, Qt.TopRightCorner)

        self.console = QTextBrowser()
        self.console.setMinimumHeight(140)
        self.console.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.console.setToolTip("Real-time logs for collection and training.")
        self.console.installEventFilter(self)

        self.console_clear_btn = QPushButton("Clear", self.console)
        self.console_clear_btn.setToolTip("Clear console output.")
        self.console_clear_btn.setFixedHeight(24)
        self.console_clear_btn.clicked.connect(self.console.clear)
        self.position_console_button()

        console_container = QWidget()
        console_layout = QVBoxLayout()
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(4)
        console_layout.addWidget(self.console)
        console_container.setLayout(console_layout)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(console_container)
        splitter.setSizes([600, 200])

        layout.addWidget(splitter)
        self.setLayout(layout)

        # connect autosave
        self.data_yaml_input.editingFinished.connect(self.persist_train_params)
        self.epochs_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.imgsz_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.batch_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.project_name_input.textEdited.connect(lambda _: self.persist_train_params())
        self.continue_checkbox.toggled.connect(lambda _: self.persist_train_params())
        self.exist_ok_checkbox.toggled.connect(lambda _: self.persist_train_params())
        self.save_period_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.patience_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.lr0_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.optimizer_input.textEdited.connect(lambda _: self.persist_train_params())
        self.mosaic_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.mixup_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.copy_paste_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.hsv_h_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.hsv_s_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.hsv_v_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.fliplr_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.flipud_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.scale_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.translate_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.shear_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.erasing_input.valueChanged.connect(lambda _: self.persist_train_params())
        self.amp_checkbox.toggled.connect(lambda _: self.persist_train_params())
        self.plots_checkbox.toggled.connect(lambda _: self.persist_train_params())
        self.save_json_checkbox.toggled.connect(lambda _: self.persist_train_params())

    def init_dataset_tab(self):
        layout = QHBoxLayout()

        left_col = QVBoxLayout()
        class_group = QGroupBox("Classes")
        class_layout = QVBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Enter class name")
        self.class_input.setToolTip("Add a class name to collect/label.")
        class_layout.addWidget(self.class_input)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add")
        btn_add.setToolTip("Add class to list.")
        btn_add.clicked.connect(self.add_class)
        btn_rem = QPushButton("Remove")
        btn_rem.setToolTip("Remove selected class.")
        btn_rem.clicked.connect(self.remove_class)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_rem)
        class_layout.addLayout(btn_row)

        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(140)
        self.class_list.setToolTip("Classes used for collection/labeling.")
        class_layout.addWidget(self.class_list)
        class_group.setLayout(class_layout)
        self.update_class_list()

        capture_group = QGroupBox("Capture")
        capture_layout = QVBoxLayout()
        self.btn_data_toggle = QPushButton("Start Data Collection")
        self.btn_data_toggle.setToolTip(
            "Start/stop semi-auto data collection from the screen."
        )
        self.btn_data_toggle.clicked.connect(self.toggle_data_collection)
        capture_layout.addWidget(self.btn_data_toggle)
        self.disable_capture_window_checkbox = QCheckBox("Disable extra preview window (OpenCV)")
        show_window = bool(self.config.get("ui", {}).get("show_capture_window", False))
        self.disable_capture_window_checkbox.setChecked(not show_window)
        self.disable_capture_window_checkbox.setToolTip(
            "If checked, only the built-in Preview panel is used."
        )
        self.disable_capture_window_checkbox.toggled.connect(self.on_capture_window_toggle)
        capture_layout.addWidget(self.disable_capture_window_checkbox)
        capture_group.setLayout(capture_layout)

        label_group = QGroupBox("Label Verification")
        label_layout = QVBoxLayout()
        btn_label_verify = QPushButton("Open Label Verification")
        btn_label_verify.setToolTip("Open label verification tool for label_data_folder.")
        btn_label_verify.clicked.connect(self.open_label_verification)
        label_layout.addWidget(btn_label_verify)
        label_group.setLayout(label_layout)

        left_col.addWidget(class_group)
        left_col.addWidget(capture_group)
        left_col.addWidget(label_group)
        left_col.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left_col)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("No preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(280, 280)
        self.preview_label.setStyleSheet("border: 1px solid #555; background-color: #111;")
        self.preview_label.setToolTip("Live frame preview during collection.")
        preview_layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)
        preview_group.setLayout(preview_layout)
        preview_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout.addWidget(left_widget, 1)
        layout.addWidget(preview_group, 0)
        layout.setAlignment(left_widget, Qt.AlignTop)
        layout.setAlignment(preview_group, Qt.AlignTop)
        self.tab_dataset.setLayout(layout)

    def init_training_tab(self):
        layout = QVBoxLayout()

        data_layout = QHBoxLayout()
        self.data_yaml_input = QLineEdit()
        self.data_yaml_input.setToolTip("Path to data.yaml (train/val).")
        last_yaml = self.config.get("last_data_yaml", "")
        self.data_yaml_input.setText(last_yaml)
        btn_browse_yaml = QPushButton("Browse...")
        btn_browse_yaml.setToolTip("Browse for data.yaml.")
        btn_browse_yaml.clicked.connect(self.browse_yaml_file)
        data_layout.addWidget(QLabel("Data yaml:"))
        data_layout.addWidget(self.data_yaml_input)
        data_layout.addWidget(btn_browse_yaml)
        layout.addLayout(data_layout)

        grp = QGroupBox("Train YOLO Parameters")
        form = QGridLayout()
        form.setHorizontalSpacing(16)
        td = self.config.get("train_defaults", {})

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100000)
        self.epochs_input.setValue(int(td.get("epochs", 50)))
        self.epochs_input.setToolTip("Number of training epochs.")
        form.addWidget(QLabel("Epochs:"), 0, 0)
        form.addWidget(self.epochs_input, 0, 1)

        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(32, 4096)
        self.imgsz_input.setSingleStep(32)
        self.imgsz_input.setValue(int(td.get("imgsz", 640)))
        self.imgsz_input.setToolTip("Image size (imgsz) for training.")
        form.addWidget(QLabel("Imgsz:"), 0, 2)
        form.addWidget(self.imgsz_input, 0, 3)

        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 4096)
        self.batch_input.setValue(int(td.get("batch", 16)))
        self.batch_input.setToolTip("Batch size for training.")
        form.addWidget(QLabel("Batch:"), 1, 0)
        form.addWidget(self.batch_input, 1, 1)

        self.project_name_input = QLineEdit()
        self.project_name_input.setText(td.get("project_name", ""))
        self.project_name_input.setToolTip("Project/name for outputs (runs/... ).")
        form.addWidget(QLabel("Project Name:"), 1, 2)
        form.addWidget(self.project_name_input, 1, 3)

        self.continue_checkbox = QCheckBox("Continue Training (resume)")
        self.continue_checkbox.setChecked(bool(td.get("resume", False)))
        self.continue_checkbox.setToolTip("Resume training from last checkpoint.")
        form.addWidget(self.continue_checkbox, 2, 0, 1, 2)

        self.exist_ok_checkbox = QCheckBox("Overwrite run folder if exists (exist_ok)")
        self.exist_ok_checkbox.setChecked(bool(td.get("exist_ok", True)))
        self.exist_ok_checkbox.setToolTip("Allow overwriting existing run folder.")
        form.addWidget(self.exist_ok_checkbox, 2, 2, 1, 2)

        self.save_period_input = QSpinBox()
        self.save_period_input.setRange(0, 200)
        self.save_period_input.setValue(int(td.get("save_period", 100)))
        self.save_period_input.setToolTip("Save weights every N epochs (0 = best only).")
        form.addWidget(QLabel("Save period:"), 3, 0)
        form.addWidget(self.save_period_input, 3, 1)

        self.patience_input = QSpinBox()
        self.patience_input.setRange(0, 100)
        self.patience_input.setValue(int(td.get("patience", 20)))
        self.patience_input.setToolTip("Early stopping patience (epochs).")
        form.addWidget(QLabel("Patience:"), 3, 2)
        form.addWidget(self.patience_input, 3, 3)

        grp.setLayout(form)
        layout.addWidget(grp)

        adv = QGroupBox("Advanced")
        adv_form = QGridLayout()
        adv_form.setHorizontalSpacing(16)

        self.lr0_input = QDoubleSpinBox()
        self.lr0_input.setRange(0.000001, 1.0)
        self.lr0_input.setDecimals(6)
        self.lr0_input.setSingleStep(0.0001)
        self.lr0_input.setValue(float(td.get("lr0", 0.001)))
        self.lr0_input.setToolTip("Base learning rate (lr0).")

        self.optimizer_input = QLineEdit()
        self.optimizer_input.setText(td.get("optimizer", "SGD"))
        self.optimizer_input.setToolTip("Optimizer name (e.g., SGD, AdamW).")

        self.mosaic_input = QDoubleSpinBox()
        self.mosaic_input.setRange(0.0, 1.0)
        self.mosaic_input.setSingleStep(0.05)
        self.mosaic_input.setValue(float(td.get("mosaic", 1.0)))
        self.mosaic_input.setToolTip("Mosaic augmentation probability.")

        self.mixup_input = QDoubleSpinBox()
        self.mixup_input.setRange(0.0, 1.0)
        self.mixup_input.setSingleStep(0.05)
        self.mixup_input.setValue(float(td.get("mixup", 0.1)))
        self.mixup_input.setToolTip("MixUp augmentation probability.")

        self.copy_paste_input = QDoubleSpinBox()
        self.copy_paste_input.setRange(0.0, 1.0)
        self.copy_paste_input.setSingleStep(0.05)
        self.copy_paste_input.setValue(float(td.get("copy_paste", 0.05)))
        self.copy_paste_input.setToolTip("Copy-paste augmentation probability.")

        self.hsv_h_input = QDoubleSpinBox()
        self.hsv_h_input.setRange(0.0, 1.0)
        self.hsv_h_input.setSingleStep(0.01)
        self.hsv_h_input.setValue(float(td.get("hsv_h", 0.02)))
        self.hsv_h_input.setToolTip("HSV hue augmentation.")

        self.hsv_s_input = QDoubleSpinBox()
        self.hsv_s_input.setRange(0.0, 1.0)
        self.hsv_s_input.setSingleStep(0.05)
        self.hsv_s_input.setValue(float(td.get("hsv_s", 0.7)))
        self.hsv_s_input.setToolTip("HSV saturation augmentation.")

        self.hsv_v_input = QDoubleSpinBox()
        self.hsv_v_input.setRange(0.0, 1.0)
        self.hsv_v_input.setSingleStep(0.05)
        self.hsv_v_input.setValue(float(td.get("hsv_v", 0.5)))
        self.hsv_v_input.setToolTip("HSV value augmentation.")

        self.fliplr_input = QDoubleSpinBox()
        self.fliplr_input.setRange(0.0, 1.0)
        self.fliplr_input.setSingleStep(0.05)
        self.fliplr_input.setValue(float(td.get("fliplr", 0.5)))
        self.fliplr_input.setToolTip("Horizontal flip probability.")

        self.flipud_input = QDoubleSpinBox()
        self.flipud_input.setRange(0.0, 1.0)
        self.flipud_input.setSingleStep(0.05)
        self.flipud_input.setValue(float(td.get("flipud", 0.0)))
        self.flipud_input.setToolTip("Vertical flip probability.")

        self.scale_input = QDoubleSpinBox()
        self.scale_input.setRange(0.0, 2.0)
        self.scale_input.setSingleStep(0.05)
        self.scale_input.setValue(float(td.get("scale", 0.6)))
        self.scale_input.setToolTip("Scale augmentation.")

        self.translate_input = QDoubleSpinBox()
        self.translate_input.setRange(0.0, 1.0)
        self.translate_input.setSingleStep(0.05)
        self.translate_input.setValue(float(td.get("translate", 0.2)))
        self.translate_input.setToolTip("Translate augmentation.")

        self.shear_input = QDoubleSpinBox()
        self.shear_input.setRange(0.0, 1.0)
        self.shear_input.setSingleStep(0.05)
        self.shear_input.setValue(float(td.get("shear", 0.1)))
        self.shear_input.setToolTip("Shear augmentation.")

        self.erasing_input = QDoubleSpinBox()
        self.erasing_input.setRange(0.0, 1.0)
        self.erasing_input.setSingleStep(0.05)
        self.erasing_input.setValue(float(td.get("erasing", 0.3)))
        self.erasing_input.setToolTip("Random erasing probability.")

        self.amp_checkbox = QCheckBox("AMP")
        self.amp_checkbox.setChecked(bool(td.get("amp", True)))
        self.amp_checkbox.setToolTip("Enable automatic mixed precision.")

        self.plots_checkbox = QCheckBox("Plots")
        self.plots_checkbox.setChecked(bool(td.get("plots", True)))
        self.plots_checkbox.setToolTip("Save training plots.")

        self.save_json_checkbox = QCheckBox("Save JSON")
        self.save_json_checkbox.setChecked(bool(td.get("save_json", True)))
        self.save_json_checkbox.setToolTip("Save COCO JSON metrics.")

        adv_form.addWidget(QLabel("lr0:"), 0, 0)
        adv_form.addWidget(self.lr0_input, 0, 1)
        adv_form.addWidget(QLabel("Optimizer:"), 0, 2)
        adv_form.addWidget(self.optimizer_input, 0, 3)

        adv_form.addWidget(QLabel("Mosaic:"), 1, 0)
        adv_form.addWidget(self.mosaic_input, 1, 1)
        adv_form.addWidget(QLabel("MixUp:"), 1, 2)
        adv_form.addWidget(self.mixup_input, 1, 3)

        adv_form.addWidget(QLabel("Copy-paste:"), 2, 0)
        adv_form.addWidget(self.copy_paste_input, 2, 1)
        adv_form.addWidget(QLabel("Scale:"), 2, 2)
        adv_form.addWidget(self.scale_input, 2, 3)

        adv_form.addWidget(QLabel("Translate:"), 3, 0)
        adv_form.addWidget(self.translate_input, 3, 1)
        adv_form.addWidget(QLabel("Shear:"), 3, 2)
        adv_form.addWidget(self.shear_input, 3, 3)

        adv_form.addWidget(QLabel("HSV H:"), 4, 0)
        adv_form.addWidget(self.hsv_h_input, 4, 1)
        adv_form.addWidget(QLabel("HSV S:"), 4, 2)
        adv_form.addWidget(self.hsv_s_input, 4, 3)

        adv_form.addWidget(QLabel("HSV V:"), 5, 0)
        adv_form.addWidget(self.hsv_v_input, 5, 1)
        adv_form.addWidget(QLabel("Flip LR:"), 5, 2)
        adv_form.addWidget(self.fliplr_input, 5, 3)

        adv_form.addWidget(QLabel("Flip UD:"), 6, 0)
        adv_form.addWidget(self.flipud_input, 6, 1)
        adv_form.addWidget(QLabel("Erasing:"), 6, 2)
        adv_form.addWidget(self.erasing_input, 6, 3)

        adv_form.addWidget(self.amp_checkbox, 7, 0)
        adv_form.addWidget(self.plots_checkbox, 7, 1)
        adv_form.addWidget(self.save_json_checkbox, 7, 2)

        adv.setLayout(adv_form)
        layout.addWidget(adv)

        self.btn_train_toggle = QPushButton("Start Training")
        self.btn_train_toggle.setToolTip("Start/stop YOLO training.")
        self.btn_train_toggle.clicked.connect(self.toggle_training)
        layout.addWidget(self.btn_train_toggle)
        layout.addStretch()

        self.tab_training.setLayout(layout)

    def init_tools_tab(self):
        layout = QGridLayout()
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        btn_split = QPushButton("Split Dataset")
        btn_split.setToolTip("Split dataset into train/val/test.")
        btn_split.clicked.connect(self.split_dataset)
        layout.addWidget(btn_split, 0, 0)

        btn_show_classes = QPushButton("Show Classes from Model")
        btn_show_classes.setToolTip("Show classes from current model.")
        btn_show_classes.clicked.connect(self.show_model_classes)
        layout.addWidget(btn_show_classes, 0, 1)

        btn_streamcut = QPushButton("Open StreamCut")
        btn_streamcut.setToolTip("Open StreamCut for VOD download/slicing.")
        btn_streamcut.clicked.connect(self.open_streamcut_dialog)
        layout.addWidget(btn_streamcut, 1, 0)

        btn_benchmark = QPushButton("Benchmark")
        btn_benchmark.setToolTip("Run ONNX benchmark (onnxruntime required).")
        btn_benchmark.clicked.connect(self.open_benchmark_dialog)
        layout.addWidget(btn_benchmark, 1, 1)

        btn_cfg = QPushButton("Config Settings")
        btn_cfg.setToolTip("Open config settings.")
        btn_cfg.clicked.connect(self.open_config_dialog)
        layout.addWidget(btn_cfg, 2, 0)

        self.btn_def = QPushButton("Restore Default Config")
        self.btn_def.setToolTip("Restore default configuration.")
        self.btn_def.clicked.connect(self.restore_default_config)
        layout.addWidget(self.btn_def, 2, 1)

        model_group = QGroupBox("YOLO Model Download")
        model_layout = QVBoxLayout()

        self.model_download_dir = QLineEdit(str(Path(__file__).resolve().parent / "models"))
        self.model_download_dir.setToolTip("Destination folder for downloaded models.")
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.model_download_dir)
        btn_dir = QPushButton("...")
        btn_dir.setFixedWidth(32)
        btn_dir.clicked.connect(self.browse_model_download_folder)
        dir_layout.addWidget(btn_dir)
        model_layout.addWidget(QLabel("Folder:"))
        model_layout.addLayout(dir_layout)

        self.model_size_tabs = QTabWidget()
        self.model_versions = ["v8", "v10", "v11", "v12", "v26"]
        size_tabs = [
            ("N", "n"),
            ("S", "s"),
            ("M", "m"),
            ("L", "l"),
            ("X", "x"),
        ]
        for title, size_code in size_tabs:
            tab = QWidget()
            grid = QGridLayout()
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(6)
            for idx, version in enumerate(self.model_versions):
                model_name = self.get_model_name(version, size_code)
                btn = QPushButton(model_name)
                btn.setToolTip(f"Download {model_name}")
                btn.clicked.connect(lambda _, v=version, s=size_code: self.download_yolo_variant(v, s))
                row = idx // 2
                col = idx % 2
                grid.addWidget(btn, row, col)
            tab.setLayout(grid)
            self.model_size_tabs.addTab(tab, title)
        model_layout.addWidget(self.model_size_tabs)

        custom_layout = QHBoxLayout()
        self.model_custom_input = QLineEdit()
        self.model_custom_input.setPlaceholderText("Custom model name or URL")
        self.model_custom_input.setToolTip("Example: yolov8n.pt or https://.../model.pt")
        btn_custom = QPushButton("Download")
        btn_custom.setToolTip("Download custom model or URL.")
        btn_custom.clicked.connect(self.download_yolo_model)
        custom_layout.addWidget(self.model_custom_input)
        custom_layout.addWidget(btn_custom)
        model_layout.addWidget(QLabel("Custom:"))
        model_layout.addLayout(custom_layout)

        note = QLabel("YOLO v26 weights may be unavailable; use a direct URL if needed.")
        note.setWordWrap(True)
        model_layout.addWidget(note)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group, 3, 1)

        theme_group = QGroupBox("Theme")
        theme_layout = QFormLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.setToolTip("Switch qt-material theme.")
        themes = self.get_material_themes()
        if themes:
            self.theme_combo.addItems(themes)
        else:
            self.theme_combo.addItem("dark_teal.xml")
            self.theme_combo.setEnabled(False)
        current_theme = self.config.get("ui", {}).get("theme", "dark_teal.xml")
        if current_theme in [self.theme_combo.itemText(i) for i in range(self.theme_combo.count())]:
            self.theme_combo.setCurrentText(current_theme)
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        theme_layout.addRow("Theme:", self.theme_combo)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group, 3, 0)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(4, 1)
        self.tab_tools.setLayout(layout)

    # -------- Helpers ----------
    def log_to_console(self, message):
        self.console.append(message)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def position_console_button(self):
        if not hasattr(self, "console_clear_btn"):
            return
        margin = 2
        btn = self.console_clear_btn
        btn.adjustSize()
        x = max(margin, self.console.viewport().width() - btn.width() - margin)
        y = margin
        btn.move(x, y)
        btn.raise_()

    def eventFilter(self, obj, event):
        if obj == getattr(self, "console", None) and event.type() == QEvent.Resize:
            self.position_console_button()
        return super().eventFilter(obj, event)

    def apply_ui_font(self, size):
        app = QApplication.instance()
        if not app:
            return
        try:
            base = app.property("base_stylesheet")
            if base:
                app.setStyleSheet(f"{base}\n*{{font-size:{int(size)}pt;}}")
                self.position_console_button()
            else:
                font = app.font()
                font.setPointSize(int(size))
                app.setFont(font)
        except Exception:
            pass

    def get_material_themes(self):
        try:
            import qt_material
            return qt_material.list_themes()
        except Exception:
            return []

    def on_theme_changed(self, theme):
        self.config.setdefault("ui", {})["theme"] = theme
        self.save_config()
        app = QApplication.instance()
        if not app:
            return
        if not apply_material_theme(app, theme=theme):
            apply_dark_theme(app)
        self.apply_ui_font(self.config.get("ui", {}).get("font_size", 11))

    def open_label_verification(self):
        script_path = Path(__file__).resolve().parent / "Core" / "labelConfig.py"
        if not script_path.exists():
            self.console.append("[ERROR] labelConfig.py not found.")
            return
        try:
            if "label_data_folder" not in self.config:
                self.config["label_data_folder"] = self.config.get("data_folder", "")
                self.save_config()
            subprocess.Popen([sys.executable, str(script_path)])
            self.console.append("[INFO] Label verification opened.")
        except Exception as e:
            self.console.append(f"[ERROR] Failed to open label verification: {e}")

    def browse_model_download_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.model_download_dir.text().strip())
        if folder:
            self.model_download_dir.setText(folder)

    def get_model_name(self, version, size_code):
        prefixes = {
            "v8": "yolov8",
            "v10": "yolov10",
            "v11": "yolo11",
            "v12": "yolo12",
            "v26": "yolo26",
        }
        prefix = prefixes.get(version, "yolov8")
        return f"{prefix}{size_code}.pt"

    def download_yolo_variant(self, version, size_code):
        model_name = self.get_model_name(version, size_code)
        self.download_yolo_model(model_name)

    def download_yolo_model(self, model_ref=None):
        if model_ref is None:
            model_ref = self.model_custom_input.text().strip()
        dest_folder = self.model_download_dir.text().strip()
        if not model_ref:
            self.console.append("[ERROR] Model name or URL is empty.")
            return
        if not dest_folder:
            self.console.append("[ERROR] Destination folder is empty.")
            return
        if hasattr(self, "model_download_thread") and self.model_download_thread.isRunning():
            self.console.append("[INFO] Model download already in progress.")
            return
        self.model_download_thread = ModelDownloadThread(model_ref, dest_folder)
        self.model_download_thread.log_signal.connect(self.log_to_console)
        self.model_download_thread.finished.connect(self.on_model_download_finished)
        self.model_download_thread.start()

    def on_model_download_finished(self, ok, path):
        if ok:
            if path:
                self.console.append(f"[INFO] Model ready: {path}")
            else:
                self.console.append("[INFO] Model download finished.")
        else:
            self.console.append("[ERROR] Model download failed.")

    def on_font_size_changed(self, size):
        self.config.setdefault("ui", {})["font_size"] = int(size)
        self.save_config()
        self.apply_ui_font(size)

    def update_class_list(self):
        self.class_list.clear()
        for i, name in enumerate(self.config.get("classes", [])):
            self.class_list.addItem(f"{i}: {name}")

    # -------- Class controls ----------
    def add_class(self):
        name = self.class_input.text().strip()
        if name and name not in self.config.setdefault("classes", []):
            self.config["classes"].append(name)
            self.save_config()
            self.update_class_list()
            self.console.append(f"Class '{name}' added.")
            self.class_input.clear()
        else:
            self.console.append("Invalid or duplicate class.")

    def remove_class(self):
        itm = self.class_list.currentItem()
        if itm:
            _, name = itm.text().split(":", 1)
            name = name.strip()
            self.config["classes"].remove(name)
            self.save_config()
            self.update_class_list()
            self.console.append(f"Class '{name}' removed.")
        else:
            self.console.append("No class selected.")

    # -------- Train params persistence ----------
    def persist_train_params(self):
        # актуализируем last_data_yaml тоже
        self.config["last_data_yaml"] = self.data_yaml_input.text().strip()
        self.config["train_defaults"] = {
            "epochs": int(self.epochs_input.value()),
            "imgsz": int(self.imgsz_input.value()),
            "batch": int(self.batch_input.value()),
            "project_name": self.project_name_input.text().strip(),
            "resume": bool(self.continue_checkbox.isChecked()),
            "exist_ok": bool(self.exist_ok_checkbox.isChecked()),
            "save_period": int(self.save_period_input.value()),
            "patience": int(self.patience_input.value()),
            "lr0": float(self.lr0_input.value()),
            "optimizer": self.optimizer_input.text().strip() or "SGD",
            "mosaic": float(self.mosaic_input.value()),
            "mixup": float(self.mixup_input.value()),
            "copy_paste": float(self.copy_paste_input.value()),
            "hsv_h": float(self.hsv_h_input.value()),
            "hsv_s": float(self.hsv_s_input.value()),
            "hsv_v": float(self.hsv_v_input.value()),
            "fliplr": float(self.fliplr_input.value()),
            "flipud": float(self.flipud_input.value()),
            "scale": float(self.scale_input.value()),
            "translate": float(self.translate_input.value()),
            "shear": float(self.shear_input.value()),
            "erasing": float(self.erasing_input.value()),
            "amp": bool(self.amp_checkbox.isChecked()),
            "plots": bool(self.plots_checkbox.isChecked()),
            "save_json": bool(self.save_json_checkbox.isChecked())
        }
        self.save_config()

    # -------- Data collection ----------
    def toggle_data_collection(self):
        if not self.data_collection_active:
            started = self.start_data_collection()
            if started:
                self.data_collection_active = True
                self.btn_data_toggle.setText("Stop Data Collection")
        else:
            # остановка
            self.stop_data_collection()
            self.data_collection_active = False
            self.btn_data_toggle.setText("Start Data Collection")

    def start_data_collection(self):
        if not self.config.get("classes"):
            self.console.append("[ERROR] No classes selected for collection.")
            return False

        try:
            model = YOLO(self.config["model_path"], verbose=False)
        except Exception as e:
            self.console.append(f"[ERROR] Failed to load model: {e}")
            return False

        model_names = model.names
        self.console.append(f"[DEBUG] model.names: {model_names}")

        reverse_names = {v: k for k, v in model_names.items()}

        class_map = {}
        for class_name in self.config["classes"]:
            if class_name in reverse_names:
                class_map[class_name] = reverse_names[class_name]
            else:
                self.console.append(f"[WARNING] Class '{class_name}' not found in model.names.")
                class_map[class_name] = 99

        self.config["class_map"] = class_map
        self.console.append(f"[INFO] Class map: {class_map}")
        self.console.append(f"[INFO] Detection threshold: {self.config.get('detection_threshold')}")

        show_window = bool(self.config.get("ui", {}).get("show_capture_window", False))
        self.capture = ScreenCapture(
            config=self.config,
            output_folder=self.config["output_folder"],
            log_callback=self.log_to_console,
            show_window=show_window
        )

        self.console.append("Starting data collection…")
        self.capture_thread = CaptureThread(self.capture)
        self.capture_thread.frame_signal.connect(self.display_frame)
        self.capture_thread.log_signal.connect(self.log_to_console)
        # Если поток завершится сам (по ошибке/по стопу) — вернуть кнопку в исходное состояние
        self.capture_thread.finished.connect(self.on_data_collection_finished)
        self.capture_thread.start()
        return True

    def stop_data_collection(self):
        if self.capture:
            self.console.append("Stopping data collection…")
            self.capture.stop_flag = True
            if self.capture_thread:
                self.capture_thread.wait()
            self.capture = None
        else:
            self.console.append("Data collection was not running.")

    def on_data_collection_finished(self):
        # поток завершился — вернуть кнопку и флаг
        self.data_collection_active = False
        if hasattr(self, "btn_data_toggle"):
            self.btn_data_toggle.setText("Start Data Collection")

    def on_capture_window_toggle(self, checked):
        ui_cfg = self.config.setdefault("ui", {})
        ui_cfg["show_capture_window"] = not checked
        self.save_config()

    # -------- Training ----------
    def toggle_training(self):
        if not self.training_active:
            # Start training
            self.train_yolo()
            self.btn_train_toggle.setText("Stop Training")
            self.training_active = True
        else:
            # Stop training
            self.stop_training()
            self.btn_train_toggle.setText("Start Training")
            self.training_active = False

    def train_yolo(self):
        data_path = self.data_yaml_input.text().strip()
        if not data_path:
            self.console.append("[ERROR] Путь к data.yaml не указан.")
            return

        # Обновляем конфиг
        self.config["last_data_yaml"] = data_path
        self.save_config()

        overrides = {}
        td = self.config.get("train_defaults", {})
        for key in (
            "lr0", "optimizer", "mosaic", "mixup", "copy_paste",
            "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud",
            "scale", "translate", "shear", "erasing",
            "amp", "plots", "save_json"
        ):
            if key in td:
                overrides[key] = td[key]

        # Формируем параметры
        params = {
            "data_yaml": data_path,
            "epochs": self.epochs_input.value(),
            "imgsz": self.imgsz_input.value(),
            "batch": self.batch_input.value(),
            "project": self.project_name_input.text().strip() or "runs/train",
            "resume": self.continue_checkbox.isChecked(),
            "exist_ok": self.exist_ok_checkbox.isChecked(),
            "save_period": self.save_period_input.value(),
            "patience": self.patience_input.value(),
            "overrides": overrides
        }

        # Запускаем обучение в QThread
        self.trainer_thread = TrainerThread(self.config, params)
        self.trainer_thread.log_signal.connect(self.log_to_console)
        self.trainer_thread.finished.connect(self.training_finished)
        self.trainer_thread.start()

        self.btn_train_toggle.setText("Stop Training")
        self.training_active = True
        self.console.append("Training started...")

    def stop_training(self):
        if self.trainer_thread and self.trainer_thread.isRunning():
            self.console.append("[INFO] Interrupting training...")
            self.trainer_thread.terminate()
            self.trainer_thread.wait()
            self.console.append("[INFO] Training was forcefully stopped.")
        self.training_active = False
        self.btn_train_toggle.setText("Start Training")

    def training_finished(self, success):
        if success:
            self.console.append("[INFO] ✅ Training completed successfully.")
        else:
            self.console.append("[ERROR] ❌ Training ended with an error.")

        self.btn_train_toggle.setText("Start Training")
        self.training_active = False
        self.btn_def.setEnabled(True)

    # -------- Misc ----------
    def show_model_classes(self):
        model_path = self.config.get("model_path", "")
        if not os.path.isfile(model_path):
            self.console.append(f"[ERROR] Model not found at path: {model_path}")
            return

        try:
            model = YOLO(model_path)
            names = model.names
            self.console.append("[INFO] Model classes:")
            for i, name in names.items():
                self.console.append(f"  {i}: {name}")
        except Exception as e:
            self.console.append(f"[ERROR] Failed to load model: {e}")

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(pix)

    def browse_yaml_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML Files (*.yaml *.yml)")
        if path:
            self.data_yaml_input.setText(path)
            self.config["last_data_yaml"] = path
            self.save_config()

    def split_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", self.config.get("data_folder", ""))
        if not folder:
            self.console.append("[INFO] Split cancelled.")
            return
        self.console.append(f"Splitting dataset: {folder}")
        script_path = Path(__file__).resolve().parent / "Core" / "splitDatasetFiles.py"
        subprocess.Popen([sys.executable, str(script_path), "--data-folder", folder])

    def open_config_dialog(self):
        dlg = ConfigDialog(self.config, self.config_path)
        if dlg.exec_():
            self.update_class_list()

    def open_streamcut_dialog(self):
        dlg = StreamCutDialog(self)
        dlg.exec_()

    def open_benchmark_dialog(self):
        dlg = BenchmarkDialog(log_callback=self.log_to_console, parent=self)
        dlg.exec_()

    def restore_default_config(self):
        default = {
            "model_path": "models/sunxds_0.7.6.pt",
            "classes": [
                "player",
                "bot",
                "weapon",
                "outline",
                "dead_body",
                "hideout_target_human",
                "hideout_target_balls",
                "head",
                "smoke",
                "fire",
                "third_person"
            ],
            "class_map": {},
            "grabber": {
                "crop_size": 0.8,
                "width": 547,
                "height": 259
            },
            "output_folder": "dataset_output",
            "save_interval": 6,
            "detection_threshold": 0.4,
            "data_folder": "stream/dataset",
            "label_data_folder": "stream/dataset",
            "last_data_yaml": "datasets/data.yaml",
            "train_defaults": {
                "epochs": 26,
                "imgsz": 640,
                "batch": 16,
                "project_name": "runs/name",
                "save_period": 50,
                "patience": 20
            }
        }
        self.config = default
        self.config.setdefault("ui", {})["theme"] = "dark_teal.xml"
        self.config.setdefault("ui", {})["font_size"] = 11
        self.save_config()
        self.update_class_list()
        self.console.append("Configuration restored to default.")
        self.data_yaml_input.setText("datasets/Valorant/data.yaml")



def apply_material_theme(app, theme="dark_teal.xml", density_scale="0"):
    try:
        # Suppress qt_material import warning for Qt5
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.setLevel(logging.ERROR)
        try:
            import qt_material
            from qt_material import build_stylesheet
            from qt_material.resources import RESOURCES_PATH
        finally:
            root_logger.setLevel(prev_level)

        extra = {"density_scale": density_scale}
        stylesheet = build_stylesheet(theme=theme, extra=extra, parent="qt_material", export=True)
        if not stylesheet:
            return False

        # Register icon search path for Qt5
        try:
            from PyQt5.QtCore import QDir
            QDir.addSearchPath("icon", str(Path(RESOURCES_PATH) / "qt_material"))
        except Exception:
            pass

        # Load fonts manually (qt_material Qt6-only loader is skipped)
        try:
            from PyQt5.QtGui import QFontDatabase
            fonts_dir = Path(qt_material.__file__).resolve().parent / "fonts" / "roboto"
            for font_path in fonts_dir.glob("*.ttf"):
                QFontDatabase.addApplicationFont(str(font_path))
        except Exception:
            pass

        app.setProperty("base_stylesheet", stylesheet)
        app.setStyle("Fusion")
        app.setStyleSheet(stylesheet)
        return True
    except Exception as exc:
        logging.warning("qt_material setup failed: %s", exc)
        return False


def apply_dark_theme(app):
    stylesheet = """
        QWidget { background-color: #2e2e2e; color: white; }
        QPushButton { background-color: #444; color: white; border: 1px solid #666; padding: 5px; }
        QLineEdit, QTextBrowser, QListWidget, QSpinBox, QDoubleSpinBox {
            background-color: #333; color: white; border: 1px solid #666;
        }
        QTabWidget::pane { border: 1px solid #444; }
        QTabBar::tab {
            background-color: #3a3a3a; color: #e6e6e6;
            padding: 6px 12px; border: 1px solid #444; border-bottom: none;
        }
        QTabBar::tab:selected { background-color: #2f2f2f; color: #ffffff; }
        QTabBar::tab:hover { background-color: #4a4a4a; }
    """
    app.setProperty("base_stylesheet", stylesheet)
    app.setStyleSheet(stylesheet)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    _theme = "dark_teal.xml"
    try:
        _cfg = Path(__file__).resolve().parent / "configs" / "config.json"
        if _cfg.exists():
            _raw = None
            for enc in ("utf-8", "utf-8-sig", "cp1251"):
                try:
                    _raw = _cfg.read_text(encoding=enc)
                    break
                except Exception:
                    _raw = None
            if _raw is None:
                _raw = _cfg.read_text(encoding="utf-8", errors="ignore")
            _data = json.loads(_raw)
            _theme = _data.get("ui", {}).get("theme", _theme)
    except Exception:
        pass
    if not apply_material_theme(app, theme=_theme):
        apply_dark_theme(app)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
