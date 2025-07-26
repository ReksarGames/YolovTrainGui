<p align="right">
  <a href="README.md">🇺🇸 English</a> | <a href="README_ru.md">🇷🇺 Русский</a>
</p>

# 🚀 YDS — YOLO Dataset Studio 🎯

> Interactive toolkit for **collecting**, **labeling**, **splitting**, and **training** object detection datasets for Ultralytics YOLO (v12 and compatible). 🖥️📊

**YOLO Flow Studio** is a desktop application with a graphical interface (PyQt5) that handles the complete workflow of YOLO dataset management:

![GUI](docs/images/gui.png)

- 🤖 Semi-automatic **screen dataset collection** using a pretrained YOLO model.
- ✏️ **Manual labeling correction** tool.
- 📂 **Automatic splitting** into `train/val/test`.
- 🏋️ **Initiating training** of Ultralytics YOLO with configurable hyperparameters.
- 📈 **Real-time logging** of data collection and training progress.

## 🌟 Features

- 🎛️ **Interactive GUI:** PyQt5-based intuitive interface.
- 🧲 **Semi-Automatic Dataset Collection:** Automatically captures screen images labeled by pretrained YOLO models.
- 🖍️ **Labeling Tool:** Manual labeling corrections via intuitive mouse controls.
- 📦 **Dataset Splitting:** Automatically splits datasets into training, validation, and testing sets.
- ⚙️ **Configurable Training:** Set epochs, image size, batch size, and more.
- ⏱️ **Real-time Logging:** Immediate feedback on dataset collection and model training.

## 🛠️ Requirements

- Python 3.8+  
- PyQt5  
- OpenCV (`opencv-python`)  
- Ultralytics (`ultralytics`)  
- PyTorch (`torch`)  
- MSS (`mss`), `screeninfo`, `numpy` 

## 📥 Installation

```bash
git clone https://github.com/your-repo/YOLOv12-Dataset-Manager.git
cd YOLOv12-Dataset-Manager
pip install -r requirements.txt
```

## 🎮 Usage

### 🚦 Launching the Application

```bash
python GUI.py
```

### 🔑 Key Functionalities

- 📸 **Data Collection:**
  - Begin automatic image capture and labeling from your screen.
- 🖌️ **Label Correction:**
  - Right-click to add bounding boxes; left-click to remove bounding boxes.
- 📚 **Dataset Management:**
  - Split datasets into `train`, `val`, and `test` subsets automatically.
- 🚀 **Model Training:**
  - Adjust training parameters via GUI, then initiate training.

## ⚙️ Configuration

Adjust settings in `config.json` or via the GUI:

```json
{
  "model_path": "models/sunxds_0.7.6.pt",
  "classes": ["class_player", "class_head"],
  "grabber": { "crop_size": 0.8, "width": 640, "height": 640 },
  "output_folder": "dataset_output",
  "save_interval": 3,
  "detection_threshold": 0.5,
  "data_folder": "dataset_output",
  "last_data_yaml": "datasets/Valorant/data.yaml"
}
```

### 🧩 Steps

1. 🎥 **Data Collection (Semi-Auto)**  
Start screen capture via GUI by pressing `Start Capture`. Captured images and labels will be saved automatically. Press `Stop` to end capture.

2. 🎨 **Manual Label Correction**  
Use the labeling tool accessible via GUI or by running `labelConfig.py`.

- **Right-click** — Add bounding box of selected class.
- **Left-click** — Remove bounding box under cursor.
- **Class switching** — Via GUI class list.

3. 📁 **Dataset Splitting**  
Split dataset via GUI (`Split` button) or command line:

```bash
python splitDatasetFiles.py
```

Default output structure:

```
dataset_output_split/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

4. 🏅 **Training**  
Specify the `data.yaml` (with paths to train/val), set hyperparameters (epochs, imgsz, batch) in GUI, and press `Train`.

CLI version:

```bash
python train.py \
  --model models/yolov12s.pt \
  --data datasets/your_dataset/data.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --project runs/exp \
  --name auto
```

## 📌 Project Structure

```
YOLO-Flow-Studio
├── dataset_output/
│   ├── images/
│   └── labels/
├── models/
│   └── yolov12s.pt
├── GUI.py                  # Main GUI
├── labelConfig.py          # Manual labeling tool
├── semiauto_dataset_collector.py  # Screen capture and auto-labeling
├── splitDatasetFiles.py    # Dataset splitting script
├── train.py                # YOLO training script
├── config.json             # Configuration file
├── LICENSE                 # MIT License
└── gui.puml                # (optional)
