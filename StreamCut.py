import os
import json
import subprocess
import cv2
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ultralytics import YOLO

import logging

# Показывать в консоли только warning+ от Ultralytics
logging.getLogger('ultralytics').setLevel(logging.WARNING)

class StreamCut:
    def __init__(self, config_path: str):
        # Загружаем конфиг
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Пути из конфига
        self.raw_stream_folder   = cfg["raw_stream_folder"]
        self.chunks_folder       = cfg["chunks_folder"]
        self.output_folder       = cfg["output_folder"]

        # Папки для датасета
        self.images_folder = os.path.join(self.output_folder, "images")
        self.labels_folder = os.path.join(self.output_folder, "labels")
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.labels_folder, exist_ok=True)

        # Остальные параметры
        self.video_sources       = cfg["video_sources"]
        self.time_interval       = cfg["time_interval"]
        self.detection_threshold = cfg["detection_threshold"]
        self.model_path          = cfg["model_path"]
        self.class_map           = cfg["class_map"]

        # Параллельность
        self.max_download_workers = cfg["max_download_workers"]
        self.split_workers        = cfg["split_workers"]
        self.process_workers      = cfg["process_workers"]
        self.save_workers         = cfg["save_workers"]

        # Архив скачанных и точки возобновления
        self.download_archive = cfg["download_archive"]
        self.resume_info_file = cfg["resume_info_file"]

        # Инициализируем модель
        self.model = YOLO(self.model_path)

    def log(self, msg: str):
        print(msg)

    def download_all(self):
        """Параллельно скачиваем все видео."""
        os.makedirs(self.raw_stream_folder, exist_ok=True)

        def dl(url):
            cmd = [
                "yt-dlp",
                "-f", "best",
                "--download-archive", self.download_archive,
                "--output", os.path.join(self.raw_stream_folder, "%(id)s.%(ext)s"),
                url
            ]
            subprocess.run(cmd, check=True)
            self.log(f"[DOWNLOAD] {url}")

        with ThreadPoolExecutor(max_workers=self.max_download_workers) as ex:
            ex.map(dl, self.video_sources)

    def split_all(self):
        """Разбиваем скачанные .ts на куски."""
        os.makedirs(self.chunks_folder, exist_ok=True)

        def split_file(path: Path):
            base        = path.stem
            out_pattern = os.path.join(self.chunks_folder, f"{base}_%03d.ts")
            cmd = [
                "ffmpeg",
                "-i", str(path),
                "-c", "copy", "-map", "0",
                "-f", "segment", "-segment_time", str(self.time_interval),
                out_pattern
            ]
            subprocess.run(cmd, check=True)
            self.log(f"[SPLIT] {base}")

        files = list(Path(self.raw_stream_folder).glob("*.ts"))
        with ThreadPoolExecutor(max_workers=self.split_workers) as exe:
            exe.map(split_file, files)

    def process_all(self):
        # Убедимся, что папка для resume_info_file существует
        os.makedirs(os.path.dirname(self.resume_info_file), exist_ok=True)

        # Загрузка информации о прогрессе
        if os.path.exists(self.resume_info_file):
            with open(self.resume_info_file, 'r') as f:
                resume = json.load(f)
        else:
            resume = {"processed_chunks": []}

        resume_lock = threading.Lock()
        save_queue = Queue()

        # Сэйвер: вынимает из очереди и пишет на диск
        def saver():
            while True:
                item = save_queue.get()
                if item is None:
                    break
                frame, img_path, lbl_path, lines, img_name = item
                cv2.imwrite(img_path, frame)
                with open(lbl_path, 'w') as fw:
                    fw.write("\n".join(lines) + "\n")
                self.log(f"[INFO] Saved image and {len(lines)} labels → {img_name}")
                save_queue.task_done()

        # Стартуем сэйвер-воркеры
        savers = []
        for _ in range(self.save_workers):
            t = threading.Thread(target=saver, daemon=True)
            t.start()
            savers.append(t)

        # Обработка одного чанка
        def process_chunk(ts_path: Path):
            base = ts_path.stem

            # Пропускаем, если уже обработан
            with resume_lock:
                if base in resume["processed_chunks"]:
                    self.log(f"[SKIP] {base} уже обработан")
                    return

            cap = cv2.VideoCapture(str(ts_path))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Центровая обрезка до 640×640 (иначе ресайз)
                crop_size = 640
                h0, w0 = frame.shape[:2]
                if h0 >= crop_size and w0 >= crop_size:
                    x1 = (w0 - crop_size) // 2
                    y1 = (h0 - crop_size) // 2
                    frame_crop = frame[y1 : y1 + crop_size, x1 : x1 + crop_size]
                else:
                    frame_crop = cv2.resize(frame, (crop_size, crop_size))

                if frame_idx % self.time_interval == 0:
                    results = self.model(
                        frame_crop,
                        imgsz=(crop_size, crop_size),
                        conf=self.detection_threshold
                    )
                    lines = []
                    w, h = crop_size, crop_size

                    for r in results:
                        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                            cls_id = int(cls)
                            if cls_id in self.class_map.values():
                                x1b, y1b, x2b, y2b = box
                                cx = ((x1b + x2b) / 2) / w
                                cy = ((y1b + y2b) / 2) / h
                                bw = (x2b - x1b) / w
                                bh = (y2b - y1b) / h
                                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                    if lines:
                        img_name = f"{base}_{frame_idx:05d}.jpg"
                        lbl_name = f"{base}_{frame_idx:05d}.txt"
                        img_path = os.path.join(self.images_folder, img_name)
                        lbl_path = os.path.join(self.labels_folder, lbl_name)
                        save_queue.put((frame_crop.copy(), img_path, lbl_path, lines, img_name))
                    else:
                        self.log(f"[WARNING] No valid detections → skipping {base}_{frame_idx:05d}.jpg")

                frame_idx += 1

            cap.release()

            # Отмечаем чанк как обработанный
            with resume_lock:
                resume["processed_chunks"].append(base)
                with open(self.resume_info_file, 'w') as f:
                    json.dump(resume, f)

        # Параллельный запуск инференса по всем чанкам
        chunks = list(Path(self.chunks_folder).glob("*.ts"))
        with ThreadPoolExecutor(max_workers=self.process_workers) as exe:
            exe.map(process_chunk, chunks)

        # Ждём завершения всех сохранений и “выключаем” сэйвер-воркеры
        save_queue.join()
        for _ in savers:
            save_queue.put(None)
        for t in savers:
            t.join()

    def run(self):
        self.download_all()
        self.split_all()
        self.process_all()
        self.log("[DONE] All done!")

if __name__ == "__main__":
    sc = StreamCut("configStreamCut.json")
    sc.run()
