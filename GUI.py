import sys
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QListWidget, \
    QTextBrowser, QFileDialog, QFormLayout, QDoubleSpinBox, QSpinBox, QDialog
from PyQt5.QtCore import Qt


class ConfigDialog(QDialog):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("Configuration Settings")
        self.setGeometry(200, 200, 400, 400)

        self.config = config
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        # Model path field
        self.model_path_input = QLineEdit(self.config.get("model_path", ""))
        layout.addRow("Model Path:", self.model_path_input)

        # Browse button for Model path
        self.model_path_browse_button = QPushButton("Browse")
        self.model_path_browse_button.clicked.connect(self.browse_model_path)
        layout.addWidget(self.model_path_browse_button)

        # Grabber settings
        self.crop_size_input = QDoubleSpinBox()
        self.crop_size_input.setValue(self.config.get("grabber", {}).get("crop_size", 0.8))
        layout.addRow("Crop Size:", self.crop_size_input)

        self.width_input = QSpinBox()
        self.width_input.setValue(self.config.get("grabber", {}).get("width", 640))
        layout.addRow("Width:", self.width_input)

        self.height_input = QSpinBox()
        self.height_input.setValue(self.config.get("grabber", {}).get("height", 640))
        layout.addRow("Height:", self.height_input)

        # Output folder path
        self.output_folder_input = QLineEdit(self.config.get("output_folder", ""))
        layout.addRow("Output Folder:", self.output_folder_input)

        # Browse button for Output folder
        self.output_folder_browse_button = QPushButton("Browse")
        self.output_folder_browse_button.clicked.connect(self.browse_output_folder)
        layout.addWidget(self.output_folder_browse_button)

        # Save interval
        self.save_interval_input = QSpinBox()
        self.save_interval_input.setValue(self.config.get("save_interval", 3))
        layout.addRow("Save Interval:", self.save_interval_input)

        # Detection threshold
        self.detection_threshold_input = QDoubleSpinBox()
        self.detection_threshold_input.setValue(self.config.get("detection_threshold", 0.5))
        layout.addRow("Detection Threshold:", self.detection_threshold_input)

        # Data folder path
        self.data_folder_input = QLineEdit(self.config.get("data_folder", ""))
        layout.addRow("Data Folder:", self.data_folder_input)

        # Browse button for Data folder
        self.data_folder_browse_button = QPushButton("Browse")
        self.data_folder_browse_button.clicked.connect(self.browse_data_folder)
        layout.addWidget(self.data_folder_browse_button)

        # Save button
        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.save_config)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def browse_model_path(self):
        """Открытие диалога для выбора модели"""
        folder_selected = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pt)")
        if folder_selected[0]:
            self.model_path_input.setText(folder_selected[0])

    def browse_output_folder(self):
        """Открытие диалога для выбора папки для выходных данных"""
        folder_selected = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_selected:
            self.output_folder_input.setText(folder_selected)

    def browse_data_folder(self):
        """Открытие диалога для выбора папки данных"""
        folder_selected = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder_selected:
            self.data_folder_input.setText(folder_selected)

    def save_config(self):
        """Сохранение изменений конфигурации"""
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

        # Сохраняем конфиг в файл
        try:
            with open('config.json', 'w') as file:
                json.dump(self.config, file, indent=4)
            self.accept()  # Закрыть диалог после сохранения
        except Exception as e:
            print(f"Error saving config: {e}")


class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('YOLO Dataset Manager')
        self.setGeometry(100, 100, 800, 500)

        # Загружаем конфигурацию
        self.config = self.load_config("config.json")
        self.classes = self.config.get("classes", [])  # Извлекаем классы из конфига
        self.init_ui()

    def load_config(self, config_file):
        """Загрузка конфигурации из файла"""
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def save_config(self):
        """Сохранение текущей конфигурации в файл"""
        try:
            with open('config.json', 'w') as file:
                json.dump(self.config, file, indent=4)
            self.console.append("Configuration saved successfully!")
        except Exception as e:
            self.console.append(f"Error saving config: {e}")

    def init_ui(self):
        layout = QVBoxLayout()

        # Panel for adding classes
        self.class_label = QLabel("Select classes:")
        layout.addWidget(self.class_label)

        # Field for entering class name
        self.class_input = QLineEdit(self)
        self.class_input.setPlaceholderText("Enter class name")
        layout.addWidget(self.class_input)

        # Button to add class
        self.add_class_button = QPushButton("Add Class", self)
        self.add_class_button.clicked.connect(self.add_class)
        layout.addWidget(self.add_class_button)

        # Button to remove class
        self.remove_class_button = QPushButton("Remove Class", self)
        self.remove_class_button.clicked.connect(self.remove_class)
        layout.addWidget(self.remove_class_button)

        # Display list of classes
        self.class_list_widget = QListWidget(self)
        self.update_class_list()
        layout.addWidget(self.class_list_widget)

        # Console to output logs
        self.console = QTextBrowser(self)
        self.console.setOpenExternalLinks(True)
        layout.addWidget(self.console)

        # Buttons for actions
        self.start_data_button = QPushButton("Start Data Collection")
        self.start_data_button.clicked.connect(self.start_data_collection)
        layout.addWidget(self.start_data_button)

        self.split_dataset_button = QPushButton("Split Dataset")
        self.split_dataset_button.clicked.connect(self.split_dataset)
        layout.addWidget(self.split_dataset_button)

        self.train_yolo_button = QPushButton("Train YOLO")
        self.train_yolo_button.clicked.connect(self.train_yolo)
        layout.addWidget(self.train_yolo_button)

        # Button to open configuration settings
        self.config_button = QPushButton("Config Settings")
        self.config_button.clicked.connect(self.open_config_dialog)
        layout.addWidget(self.config_button)

        # Button to restore default config
        self.default_config_button = QPushButton("Restore Default Config")
        self.default_config_button.clicked.connect(self.restore_default_config)
        layout.addWidget(self.default_config_button)

        self.setLayout(layout)

    def update_class_list(self):
        """Обновление списка классов с индексацией"""
        self.class_list_widget.clear()
        for index, class_name in enumerate(self.classes):
            self.class_list_widget.addItem(f"{index}: {class_name}")  # Индексация классов

    def add_class(self):
        """Добавляет новый класс в список и отображает его в ListWidget"""
        class_name = self.class_input.text().strip()
        if class_name and class_name not in self.classes:
            self.classes.append(class_name)
            self.update_class_list()  # Обновляем список классов
            self.console.append(f"Class '{class_name}' added.")  # Выводим в консоль
            self.class_input.clear()  # Очищаем поле ввода

            # Автоматически сохраняем конфиг
            self.save_config()

        else:
            self.console.append("Invalid or duplicate class name.")

    def remove_class(self):
        """Удаляет выбранный класс из списка"""
        selected_item = self.class_list_widget.currentItem()  # Получаем выбранный элемент в списке
        if selected_item:
            class_name = selected_item.text().split(":")[1].strip()  # Получаем название класса
            self.classes.remove(class_name)  # Убираем класс из внутреннего списка
            self.update_class_list()  # Обновляем список классов в интерфейсе
            self.console.append(f"Class '{class_name}' removed.")  # Выводим в консоль

            # Автоматически сохраняем конфиг
            self.save_config()

        else:
            self.console.append("No class selected for removal.")  # Если класс не выбран

    def open_config_dialog(self):
        """Открытие окна настроек конфигурации"""
        dialog = ConfigDialog(self.config)
        dialog.exec_()

    def restore_default_config(self):
        """Восстановление конфигурации по умолчанию"""
        default_config = {
            "model_path": "models/sunxds_0.7.6.pt",
            "classes": ["class_player", "class_head"],
            "grabber": {
                "crop_size": 0.8,
                "width": 640,
                "height": 640
            },
            "output_folder": "dataset_output",
            "save_interval": 3,
            "detection_threshold": 0.5,
            "data_folder": "dataset_output"
        }

        self.config = default_config
        self.classes = default_config["classes"]
        self.update_class_list()  # Обновляем список классов в интерфейсе

        # Автоматически сохраняем конфиг
        self.save_config()

        self.console.append("Configuration restored to default.")

    def start_data_collection(self):
        """Запуск сбора данных (пример)"""
        self.console.append("Starting data collection...")  # Лог
        self.run_program('python collect_data.py')  # Запуск программы сбора данных (пример)

    def split_dataset(self):
        """Запуск разделения датасета (пример)"""
        self.console.append("Splitting dataset...")  # Лог
        self.run_program('python split_dataset.py')  # Запуск программы разделения данных (пример)

    def train_yolo(self):
        """Запуск тренировки модели YOLO (пример)"""
        self.console.append("Training YOLO model...")  # Лог
        self.run_program('python train_yolo.py')  # Запуск программы тренировки модели (пример)

    def run_program(self, command):
        """Запуск внешней программы с помощью subprocess"""
        import subprocess
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.console.append(result.stdout.decode())
            if result.stderr:
                self.console.append(f"Error: {result.stderr.decode()}")
        except subprocess.CalledProcessError as e:
            self.console.append(f"Error: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
