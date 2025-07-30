<p align="right">
  <a href="README.md">🇺🇸 English</a> | <a href="README_ru.md">🇷🇺 Русский</a>
</p>

# 🚀 YDS — YOLO Dataset Studio 🎯

> Интерактивный инструментарий для **сбора**, **разметки**, **сплита** и **обучения** датасетов детекции объектов на Ultralytics YOLO (v12 и совместимые). 🖥️📊

**YOLO Flow Studio** — десктоп‑приложение с графическим интерфейсом (PyQt5) для полного цикла работы с датасетами под YOLO:

[📹 StreamCut — автоматизированный инструмент для скачивания, нарезки и обработки Twitch VOD](https://github.com/ReksarGames/StreamCut)  

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

## 🧩 Этапы работы

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

```
---

# 🚀 StreamCut

Инструмент для автоматизированного скачивания, нарезки и обработки видеопотоков (Twitch VOD), с сохранением только тех кадров, где обнаружены целевые объекты (с помощью YOLO).


## 📋 Описание

1. 📥 **Скачивание** Параллельная загрузка VOD через `yt-dlp` с ведением архива, чтобы не дублировать загрузки.  
2. 🎞️ **Нарезка** Разбиение каждого видео на сегменты `.ts` фиксированной длины с помощью `ffmpeg`.  
3. 🤖 **Инференс** Запуск YOLO на каждом N‑м кадре; сохранение изображений и меток только при наличии детекций.  
4. ⚙️ **Параллельность** Настраиваемые пулы потоков для каждого этапа: скачивание, нарезка, инференс, сохранение.


## ✨ Основные возможности

- 🚀 Поддержка произвольного числа VOD — скачивайте любое количество Twitch‑ссылок.  
- 📸 Сохраняются только кадры с ненулевыми детекциями.  
- 🔄 Гибкая настройка параллельности на каждом этапе.  
- 🗂️ Архив скачанных видео в `downloaded.txt` для пропуска уже загруженных.  
- 🔄 Возобновляемая обработка через `resume.json`.  
- 📁 Итоговый датасет:  
  - `stream/dataset/images/*.jpg`  
  - `stream/dataset/labels/*.txt`  


```
{
  "video_sources": [
    "https://www.twitch.tv/videos/2522936875",
    "https://www.twitch.tv/videos/2524662899"
  ],
  "raw_stream_folder": "stream/raw_streams",
  "chunks_folder": "stream/chunks",
  "output_folder": "stream/dataset",
  "time_interval": 3,
  "detection_threshold": 0.3,
  "model_path": "models/sunxds_0.7.6.pt",
  "class_map": {
    "player": 0,
    "head": 7
  },
  "max_download_workers": 2,
  "split_workers": 4,
  "process_workers": 6,
  "save_workers": 2,
  "download_archive": "stream/downloaded.txt",
  "resume_info_file": "stream/resume.json"
}

```

- 🎯 video_sources — список URL для скачивания.
- 📁 raw_stream_folder — папка для сохранения исходных видео.
- 🧩 chunks_folder — папка для сегментов .ts.
- 🎯 output_folder — корень датасета (images/ и labels/ внутри).
- ⏱️ time_interval(sec) — через сколько кадров запускать инференс. 
- 🔥 detection_threshold — порог confidence для детекции.
- 🧠 model_path — путь до .pt модели YOLO.
- 🏷️ class_map — сопоставление названий классов и их ID в модели.
- 🧵 _*_workers — числа потоков для загрузки, нарезки, инференса и сохранения.
  - 🛠 Потоки для скачивания - max_download_workers
  - 🪓 Потоки для нарезки видео - split_workers
  - 🔍 Потоки для инференса (YOLO) - process_workers
  - 💾 Потоки для сохранения кадров/лейблов - save_workers
- 📜 download_archive — файл с архивом скачанных видео (чтобы не дублировать).

---
 
## 🔧 Параметры потоков

| Параметр               | Emoji | Описание                                              | Рекомендации                                       |
| ---------------------- | :---: | ----------------------------------------------------- | -------------------------------------------------- |
| `max_download_workers` |   🛠  | Число параллельных потоков для `yt-dlp`               | 2–4 (не перегружать сеть)                          |
| `split_workers`        |   🪓  | Потоки для нарезки `.ts` сегментов                    | По числу ядер CPU                                  |
| `process_workers`      |   🔍  | Потоки для инференса YOLO (чтение и обработка чанков) | По возможностям GPU                                |
| `save_workers`         |   💾  | Потоки для записи изображений и меток на диск         | 1–2 (чтобы не блокировать инференс операциями I/O) |

## 📂 Структура проекта
```
├── StreamCut.py          
├── configStreamCut.json           
├── stream/
│   ├── raw_streams/      # Исходные видео
│   ├── chunks/           # Сегменты .ts
│   ├── downloaded.txt    # Архив скачанных URL
│   └── resume.json       # (опционально) для возобновления
├── stream/
│   └── dataset/
│       ├── images/       # Сохранённые кадры .jpg
│       └── labels/       # YOLO‑метки .txt
└── models/
    └── sunx.pt   # Пример модели
```

## 📝 Заметки
- time_interval задаётся в кадрах. Если нужно “каждые 10 секунд”, рассчитайте interval_frames = fps * 10.
- resume.json создаётся автоматически при первом запуске.
- Подбирайте число потоков в зависимости от вашего железа:
  - CPU‑ядра → нарезка
  - GPU → инференс
  - Диск → запись