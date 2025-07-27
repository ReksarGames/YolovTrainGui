import os
import subprocess
import sys
import json

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel,
    QListWidget, QTextBrowser, QFileDialog, QFormLayout, QDoubleSpinBox,
    QSpinBox, QDialog, QGroupBox, QHBoxLayout, QCheckBox, QSizePolicy, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

from semiauto_dataset_collector import ScreenCapture
from train import train_yolo


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
                log=log
            )
            self.finished.emit(exit_code == 0)
        except Exception as e:
            log(f"[ERROR] Exception during training: {e}")
            self.finished.emit(False)

    def stop(self):
        self._stop_flag = True  # если в будущем будет поддержка принудительной остановки

from PyQt5.QtCore import pyqtSignal

class CaptureThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, capture_instance):
        super().__init__()
        self.capture_instance = capture_instance
        self.capture_instance.on_frame_ready = self.emit_frame

    def emit_frame(self, frame):
        self.frame_signal.emit(frame)

    def run(self):
        self.capture_instance.capture_and_display()


class ConfigDialog(QDialog):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("Configuration Settings")
        self.setGeometry(200, 200, 400, 400)
        self.config = config
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(640, 640)
        self.preview_label.setStyleSheet("border: 1px solid #888; background-color: #111;")
        layout.addWidget(self.preview_label)

        self.model_path_input = QLineEdit(self.config.get("model_path", ""))
        layout.addRow("Model Path:", self.model_path_input)
        btn_model = QPushButton("Browse Model…")
        btn_model.clicked.connect(self.browse_model_path)
        layout.addWidget(btn_model)

        self.crop_size_input = QDoubleSpinBox()
        self.crop_size_input.setValue(self.config.get("grabber", {}).get("crop_size", 0.8))
        layout.addRow("Crop Size:", self.crop_size_input)

        self.width_input = QSpinBox()
        self.width_input.setValue(self.config.get("grabber", {}).get("width", 640))
        layout.addRow("Width:", self.width_input)

        self.height_input = QSpinBox()
        self.height_input.setValue(self.config.get("grabber", {}).get("height", 640))
        layout.addRow("Height:", self.height_input)

        self.output_folder_input = QLineEdit(self.config.get("output_folder", ""))
        layout.addRow("Output Folder:", self.output_folder_input)
        btn_out = QPushButton("Browse Output…")
        btn_out.clicked.connect(self.browse_output_folder)
        layout.addWidget(btn_out)

        self.save_interval_input = QSpinBox()
        self.save_interval_input.setValue(self.config.get("save_interval", 3))
        layout.addRow("Save Interval:", self.save_interval_input)

        self.detection_threshold_input = QDoubleSpinBox()
        self.detection_threshold_input.setValue(self.config.get("detection_threshold", 0.5))
        layout.addRow("Detection Threshold:", self.detection_threshold_input)

        self.data_folder_input = QLineEdit(self.config.get("data_folder", ""))
        layout.addRow("Data Folder:", self.data_folder_input)
        btn_data = QPushButton("Browse Data…")
        btn_data.clicked.connect(self.browse_data_folder)
        layout.addWidget(btn_data)

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
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
            self.accept()
        except Exception as e:
            print(f"Error saving config: {e}")


class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.btn_def = None
        self.setWindowTitle('YOLO Dataset Manager')
        self.setGeometry(100, 100, 800, 500)

        self.config = self.load_config("config.json")
        self.capture = None
        self.capture_thread = None

        self.training_active = False

        self.init_ui()

    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def save_config(self):
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
            self.console.append("Configuration saved.")
        except Exception as e:
            self.console.append(f"Error saving configuration: {e}")

    def init_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Select classes:"))
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Enter class name")
        layout.addWidget(self.class_input)

        btn_add = QPushButton("Add Class")
        btn_add.clicked.connect(self.add_class)
        layout.addWidget(btn_add)

        btn_rem = QPushButton("Remove Class")
        btn_rem.clicked.connect(self.remove_class)
        layout.addWidget(btn_rem)

        splitter = QSplitter(Qt.Vertical)

        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(80)
        self.class_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        splitter.addWidget(self.class_list)
        self.update_class_list()

        self.console = QTextBrowser()
        self.console.setMinimumHeight(100)
        self.console.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        splitter.addWidget(self.console)

        splitter.setSizes([200, 300])
        layout.addWidget(splitter)

        btn_start = QPushButton("Start Data Collection")
        btn_start.clicked.connect(self.start_data_collection)
        layout.addWidget(btn_start)

        btn_stop = QPushButton("Stop Data Collection")
        btn_stop.clicked.connect(self.stop_data_collection)
        layout.addWidget(btn_stop)

        btn_split = QPushButton("Split Dataset")
        btn_split.clicked.connect(self.split_dataset)
        layout.addWidget(btn_split)

        btn_show_classes = QPushButton("Show Classes from Model")
        btn_show_classes.clicked.connect(self.show_model_classes)
        layout.addWidget(btn_show_classes)

        grp = QGroupBox("Train YOLO Parameters")
        form = QFormLayout()
        data_layout = QHBoxLayout()

        self.data_yaml_input = QLineEdit()
        last_yaml = self.config.get("last_data_yaml", "")
        self.data_yaml_input.setText(last_yaml)
        btn_browse_yaml = QPushButton("Browse…")
        btn_browse_yaml.clicked.connect(self.browse_yaml_file)
        data_layout.addWidget(self.data_yaml_input)
        data_layout.addWidget(btn_browse_yaml)
        form.addRow("Data yaml:", data_layout)

        self.epochs_input = QSpinBox(); self.epochs_input.setValue(50)
        form.addRow("Epochs:", self.epochs_input)
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(320, 640)
        self.imgsz_input.setValue(640)
        form.addRow("Imgsz:", self.imgsz_input)
        self.batch_input = QSpinBox(); self.batch_input.setValue(16)
        form.addRow("Batch:", self.batch_input)
        self.project_name_input = QLineEdit()
        form.addRow("Project Name:", self.project_name_input)
        self.continue_checkbox = QCheckBox("Continue Training (resume)")
        form.addRow("", self.continue_checkbox)

        grp.setLayout(form)
        layout.addWidget(grp)

        self.btn_train_toggle = QPushButton("Start Training")
        self.btn_train_toggle.clicked.connect(self.toggle_training)
        layout.addWidget(self.btn_train_toggle)

        btn_cfg = QPushButton("Config Settings")
        btn_cfg.clicked.connect(self.open_config_dialog)
        layout.addWidget(btn_cfg)

        btn_def = QPushButton("Restore Default Config")
        btn_def.clicked.connect(self.restore_default_config)
        layout.addWidget(btn_def)

        self.setLayout(layout)

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

    def update_class_list(self):
        self.class_list.clear()
        for i, name in enumerate(self.config.get("classes", [])):
            self.class_list.addItem(f"{i}: {name}")

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

    def log_to_console(self, message):
        self.console.append(message)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def start_data_collection(self):
        if not self.config.get("classes"):
            self.console.append("[ERROR] No classes selected for collection.")
            return

        # Загружаем модель YOLO
        model = YOLO(self.config["model_path"], verbose=False)
        model_names = model.names  # {0: 'class_player', 1: 'class_head', ...}
        self.console.append(f"[DEBUG] model.names: {model_names}")

        # Создаём reverse-словарь { 'class_player': 0, ... }
        reverse_names = {v: k for k, v in model_names.items()}

        # Строим class_map для всех классов из config["classes"]
        class_map = {}
        for class_name in self.config["classes"]:
            if class_name in reverse_names:
                class_map[class_name] = reverse_names[class_name]
            else:
                self.console.append(f"[WARNING] Class '{class_name}' not found in model.names.")
                class_map[class_name] = 99  # 99 = игнорируемый класс

        self.config["class_map"] = class_map
        self.console.append(f"[INFO] Class map: {class_map}")
        self.console.append(f"[INFO] Detection threshold: {self.config.get('detection_threshold')}")

        # Пересоздаём ScreenCapture каждый раз с актуальным конфигом
        self.capture = ScreenCapture(
            config=self.config,
            output_folder=self.config["output_folder"],
            log_callback=self.log_to_console
        )

        self.console.append("Starting data collection…")
        self.capture_thread = CaptureThread(self.capture)
        self.capture_thread.frame_signal.connect(self.display_frame)
        self.capture_thread.start()

    def stop_data_collection(self):
        if self.capture:
            self.console.append("Stopping data collection…")
            self.capture.stop_flag = True
            if self.capture_thread:
                self.capture_thread.wait()
            self.capture = None
        else:
            self.console.append("Data collection was not running.")

    def browse_yaml_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML Files (*.yaml *.yml)")
        if path:
            self.data_yaml_input.setText(path)
            self.config["last_data_yaml"] = path
            self.save_config()

    def split_dataset(self):
        self.console.append("Splitting dataset…")
        os.system('python splitDatasetFiles.py')

    def stop_training(self):
        if self.trainer_thread and self.trainer_thread.isRunning():
            self.console.append("[INFO] Interrupting training...")
            self.trainer_thread.terminate()
            self.trainer_thread.wait()
            self.console.append("[INFO] Training was forcefully stopped.")
        self.training_active = False
        self.btn_train_toggle.setText("Start Training")

    def train_yolo(self):
        data_path = self.data_yaml_input.text().strip()
        if not data_path:
            self.console.append("[ERROR] Путь к data.yaml не указан.")
            return

        # Обновляем конфиг
        self.config["last_data_yaml"] = data_path
        self.save_config()

        # Формируем параметры
        params = {
            "data_yaml": data_path,
            "epochs": self.epochs_input.value(),
            "imgsz": self.imgsz_input.value(),
            "batch": self.batch_input.value(),
            "project": self.project_name_input.text().strip() or "runs/train",
            "resume": self.continue_checkbox.isChecked()
        }

        # Запускаем обучение в QThread
        self.trainer_thread = TrainerThread(self.config, params)
        self.trainer_thread.log_signal.connect(self.log_to_console)
        self.trainer_thread.finished.connect(self.training_finished)
        self.trainer_thread.start()

        self.btn_train_toggle.setText("Stop Training")
        self.training_active = True
        self.console.append("Training started...")

    def training_finished(self, success):
        if success:
            self.console.append("[INFO] ✅ Training completed successfully.")
        else:
            self.console.append("[ERROR] ❌ Training ended with an error.")

        self.btn_train_toggle.setText("Start Training")
        self.training_active = False
        self.btn_def.setEnabled(True)

    def open_config_dialog(self):
        dlg = ConfigDialog(self.config)
        if dlg.exec_():
            self.update_class_list()

    def restore_default_config(self):
        default = {
            "model_path": "models/sunxds_0.7.6.pt",
            "classes": ["class_player", "class_head"],
            "class_map": {
                "class_player": 0,
                "class_bot": 99,
                "class_weapon": 99,
                "class_outline": 99,
                "class_dead_body": 99,
                "class_hideout_target_human": 99,
                "class_hideout_target_balls": 99,
                "class_head": 7,
                "class_smoke": 99,
                "class_fire": 99,
                "class_third_person": 99
            },
            "grabber": {
                "crop_size": 0.8,
                "width": 640,
                "height": 640
            },
            "output_folder": "dataset_output",
            "save_interval": 3,
            "detection_threshold": 0.35,
            "data_folder": "dataset_output",
            "last_data_yaml": "E:/Project/PythonProjects/YolovTrainGui/datasets/Valorant/data - Copy.yaml"
        }
        self.config = default
        self.save_config()
        self.update_class_list()
        self.console.append("Configuration restored to default.")
        self.data_yaml_input.setText("datasets/Valorant/data.yaml")


def apply_dark_theme(app):
    app.setStyleSheet("""
        QWidget { background-color: #2e2e2e; color: white; }
        QPushButton { background-color: #444; color: white; border: 1px solid #666; padding: 5px; }
        QLineEdit, QTextBrowser, QListWidget, QSpinBox, QDoubleSpinBox {
            background-color: #333; color: white; border: 1px solid #666;
        }
    """)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
