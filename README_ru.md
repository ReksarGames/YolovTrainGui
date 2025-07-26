<p align="right">
  <a href="README.md">🇺🇸 English</a> | <a href="README_ru.md">🇷🇺 Русский</a>
</p>

# 🚀 YDS — YOLO Dataset Studio 🎯

> Интерактивный инструментарий для **сбора**, **разметки**, **сплита** и **обучения** датасетов детекции объектов на Ultralytics YOLO (v12 и совместимые). 🖥️📊

**YOLO Flow Studio** — десктоп‑приложение с графическим интерфейсом (PyQt5) для полного цикла работы с датасетами под YOLO:

![GUI](docs/images/gui.png)

- 🤖 Полуавтоматический **сбор датасета со экрана** с предварительно обученной моделью YOLO.
- ✏️ **Ручная корректировка** разметки.
- 📂 **Автоматический сплит** на `train/val/test`.
- 🏋️ **Запуск обучения** Ultralytics YOLO с управляемыми гиперпараметрами.
- 📈 **Онлайн‑логирование** хода сбора и обучения.

## 🌟 Особенности

- 🎛️ **Интерактивный GUI:** удобный интерфейс на PyQt5.
- 🧲 **Полуавтоматический сбор данных:** авто-захват и разметка изображений с экрана.
- 🖍️ **Инструмент разметки:** ручная корректировка с помощью мыши.
- 📦 **Сплит датасета:** автоматическое разделение на train, val и test.
- ⚙️ **Настройка обучения:** конфигурация параметров (эпохи, размер изображений, batch).
- ⏱️ **Онлайн-логирование:** мониторинг сбора данных и обучения в реальном времени.

## 🛠️ Требования

- Python 3.8+  
- PyQt5  
- OpenCV (`opencv-python`)  
- Ultralytics (`ultralytics`)  
- PyTorch (`torch`)  
- MSS (`mss`), `screeninfo`, `numpy`

## 📥 Установка

```bash
git clone https://github.com/your-repo/YOLOv12-Dataset-Manager.git
cd YOLOv12-Dataset-Manager
pip install -r requirements.txt
```

## 🎮 Использование

### 🚦 Запуск приложения

```bash
python GUI.py
```

### 🔑 Основные функции

- 📸 **Сбор данных:**
  - Начните автоматический захват и разметку изображений с экрана.
- 🖌️ **Корректировка разметки:**
  - ПКМ для добавления бокса; ЛКМ для удаления бокса.
- 📚 **Управление датасетом:**
  - Сплит на `train`, `val`, и `test` одним нажатием.
- 🚀 **Обучение модели:**
  - Настройка гиперпараметров и запуск обучения прямо в GUI.

## ⚙️ Конфигурация

Настройки находятся в файле `config.json` или доступны в GUI:

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

### 🧩 Этапы работы

1. 🎥 **Сбор данных (Semi-Auto)**  
В GUI нажмите `Start Capture`. Изображения и метки будут сохранены автоматически. Для остановки нажмите `Stop`.

2. 🎨 **Ручная корректировка разметки**  
Запустите инструмент разметки через GUI или `labelConfig.py`.

- **ПКМ** — добавить bounding box.
- **ЛКМ** — удалить bounding box.
- Переключение классов через список в GUI.

3. 📁 **Сплит датасета**  
Кнопка в GUI (`Split`) или команда в консоли:

```bash
python splitDatasetFiles.py
```

Структура по умолчанию:

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

4. 🏅 **Обучение**  
Укажите путь к `data.yaml`, гиперпараметры (epochs, imgsz, batch) в GUI и нажмите `Train`.

CLI-вариант:

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

## 📌 Структура проекта

```
YOLO-Flow-Studio
├── dataset_output/
│   ├── images/
│   └── labels/
├── models/
│   └── yolov12s.pt
├── GUI.py                  # Основной интерфейс
├── labelConfig.py          # Ручная разметка
├── semiauto_dataset_collector.py  # Захват экрана и авто-разметка
├── splitDatasetFiles.py    # Скрипт сплита датасета
├── train.py                # Скрипт обучения YOLO
├── config.json             # Файл конфигурации
├── LICENSE                 # Лицензия MIT
└── gui.puml                # (опционально)
