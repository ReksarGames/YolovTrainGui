import os
import sys
import json

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel,
    QListWidget, QTextBrowser, QFileDialog, QFormLayout, QDoubleSpinBox,
    QSpinBox, QDialog, QGroupBox
)
from PyQt5.QtCore import Qt, QThread

from semiauto_dataset_collector import ScreenCapture


class CaptureThread(QThread):
    def __init__(self, capture_instance):
        super().__init__()
        self.capture_instance = capture_instance

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
        self.setWindowTitle('YOLO Dataset Manager')
        self.setGeometry(100, 100, 800, 500)

        self.config = self.load_config("config.json")
        self.capture = None
        self.capture_thread = None

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

        self.class_list = QListWidget()
        layout.addWidget(self.class_list)
        self.update_class_list()

        self.console = QTextBrowser()
        layout.addWidget(self.console)

        btn_start = QPushButton("Start Data Collection")
        btn_start.clicked.connect(self.start_data_collection)
        layout.addWidget(btn_start)

        btn_stop = QPushButton("Stop Data Collection")
        btn_stop.clicked.connect(self.stop_data_collection)
        layout.addWidget(btn_stop)

        btn_split = QPushButton("Split Dataset")
        btn_split.clicked.connect(self.split_dataset)
        layout.addWidget(btn_split)

        grp = QGroupBox("Train YOLO Parameters")
        form = QFormLayout()
        self.data_yaml_input = QLineEdit()
        form.addRow("Data yaml:", self.data_yaml_input)
        self.epochs_input = QSpinBox(); self.epochs_input.setValue(50)
        form.addRow("Epochs:", self.epochs_input)
        self.imgsz_input = QSpinBox(); self.imgsz_input.setValue(640)
        form.addRow("Imgsz:", self.imgsz_input)
        self.batch_input = QSpinBox(); self.batch_input.setValue(16)
        form.addRow("Batch:", self.batch_input)
        self.project_name_input = QLineEdit()
        form.addRow("Project Name:", self.project_name_input)
        grp.setLayout(form)
        layout.addWidget(grp)

        btn_train = QPushButton("Train YOLO")
        btn_train.clicked.connect(self.train_yolo)
        layout.addWidget(btn_train)

        btn_cfg = QPushButton("Config Settings")
        btn_cfg.clicked.connect(self.open_config_dialog)
        layout.addWidget(btn_cfg)

        btn_def = QPushButton("Restore Default Config")
        btn_def.clicked.connect(self.restore_default_config)
        layout.addWidget(btn_def)

        self.setLayout(layout)

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

    def start_data_collection(self):
        if self.capture is None:
            self.capture = ScreenCapture(
                config=self.config,
                output_folder=self.config["output_folder"]
            )
        self.console.append("Starting data collection…")
        self.capture_thread = CaptureThread(self.capture)
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

    def split_dataset(self):
        self.console.append("Splitting dataset…")
        os.system('python splitDatasetFiles.py')

    def train_yolo(self):
        self.console.append("Training YOLO model…")
        cmd = (
            f'python train.py --data {self.data_yaml_input.text()} '
            f'--epochs {self.epochs_input.value()} '
            f'--img-size {self.imgsz_input.value()} '
            f'--batch-size {self.batch_input.value()} '
            f'--project {self.project_name_input.text()}'
        )
        os.system(cmd)

    def open_config_dialog(self):
        dlg = ConfigDialog(self.config)
        if dlg.exec_():
            self.update_class_list()

    def restore_default_config(self):
        default = {
            "model_path": "models/sunxds_0.7.6.pt",
            "classes": ["class_player", "class_head"],
            "grabber": {"crop_size": 0.8, "width": 640, "height": 640},
            "output_folder": "dataset_output",
            "save_interval": 3,
            "detection_threshold": 0.5,
            "data_folder": "dataset_output"
        }
        self.config = default
        self.save_config()
        self.update_class_list()
        self.console.append("Configuration restored to default.")


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
