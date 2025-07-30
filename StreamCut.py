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
        # ——— Загрузка конфига ———
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # ——— Папки ———
        self.raw_stream_folder   = cfg["raw_stream_folder"]
        self.chunks_folder       = cfg["chunks_folder"]
        self.output_folder       = cfg["output_folder"]

        os.makedirs(self.raw_stream_folder, exist_ok=True)
        os.makedirs(self.chunks_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        self.images_folder = os.path.join(self.output_folder, "images")
        self.labels_folder = os.path.join(self.output_folder, "labels")
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.labels_folder, exist_ok=True)

        # ——— Параметры ———
        self.video_sources       = cfg["video_sources"]
        self.time_interval       = cfg["time_interval"]        # seconds per segment
        self.chunks_per_stream   = cfg["chunks_per_stream"]    # how many segments per video
        self.detection_threshold = cfg["detection_threshold"]
        self.model_path          = cfg["model_path"]
        self.class_map           = cfg["class_map"]

        # ——— Параллельность ———
        self.max_download_workers = cfg["max_download_workers"]
        self.split_workers        = cfg["split_workers"]
        self.process_workers      = cfg["process_workers"]
        self.save_workers         = cfg["save_workers"]

        # ——— Архив скачанных и резюме ———
        self.download_archive = cfg["download_archive"]
        self.resume_info_file = cfg["resume_info_file"]
        self.ffmpeg_path = cfg["ffmpeg_path"]

        # ——— Модель ———
        self.model = YOLO(self.model_path)

    def log(self, msg: str):
        print(msg)

    def download_all(self):
        """Параллельно скачиваем VOD в raw_stream_folder."""
        def dl(url):
            cmd = [
                "yt-dlp",
                "--download-archive", self.download_archive,
                "--output", os.path.join(self.raw_stream_folder, "%(id)s.%(ext)s"),
                url
            ]
            subprocess.run(cmd, check=True)
            self.log(f"[DOWNLOAD] {url}")

        with ThreadPoolExecutor(max_workers=self.max_download_workers) as ex:
            ex.map(dl, self.video_sources)

    def split_all(self):
        """
        Разбиваем скачанные видео на .ts-чанки.
        Каждый файл режется на self.chunks_per_stream равных сегментов,
        покрывающих всю его длину. Если уже есть хотя бы self.chunks_per_stream
        чанков для данного видео, нарезка пропускается.
        """
        os.makedirs(self.chunks_folder, exist_ok=True)
        files = list(Path(self.raw_stream_folder).glob("*.*"))
        if not files:
            self.log(f"[SPLIT] ⚠ Нет исходных видео в {self.raw_stream_folder}")
            return

        self.log(f"[SPLIT] Режу {len(files)} файлов на {self.chunks_per_stream} частей…")

        def split_file(path: Path):
            base = path.stem

            # проверяем, сколько уже есть чанков
            existing = list(Path(self.chunks_folder).glob(f"{base}_*.ts"))
            if len(existing) >= self.chunks_per_stream:
                self.log(f"[SPLIT] {base} — найдено {len(existing)}/{self.chunks_per_stream} чанков, пропускаем")
                return

            try:
                duration = self._get_duration(path)
            except Exception as e:
                self.log(f"[ERROR] Не удалось узнать длительность {base}: {e}")
                return

            seg_time = duration / self.chunks_per_stream
            out_pattern = os.path.join(self.chunks_folder, f"{base}_%03d.ts")
            cmd = [
                self.ffmpeg_path, "-y",
                "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-c", "copy", "-map", "0",
                "-f", "segment",
                "-segment_time", str(seg_time),
                "-segment_format", "mpegts",
                out_pattern
            ]
            try:
                subprocess.run(cmd, check=True)
                self.log(f"[SPLIT] {base} → {self.chunks_per_stream}×{seg_time:.1f}s")
            except subprocess.CalledProcessError as e:
                self.log(f"[ERROR] FFmpeg нарезка {base} упала: {e}")

        with ThreadPoolExecutor(max_workers=self.split_workers) as ex:
            ex.map(split_file, files)


    def _get_duration(self, path: Path) -> float:
        """Возвращает длительность видео в секундах через ffprobe."""
        # Берём папку от ffmpeg и меняем имя на ffprobe.exe
        ffmpeg = Path(self.ffmpeg_path)
        ffprobe = ffmpeg.with_name("ffprobe.exe")

        cmd = [
            str(ffprobe),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return float(res.stdout)

    def process_all(self):
        """
        Обходим .ts-чанки, делаем инференс и сохраняем кадры,
        с прогрессом в resume_info_file, и центрированным 640×640 crop.
        """
        # — Подготовка resume-файла —
        os.makedirs(os.path.dirname(self.resume_info_file), exist_ok=True)
        if os.path.exists(self.resume_info_file):
            with open(self.resume_info_file, 'r') as f:
                resume = json.load(f)
        else:
            resume = {"processed_chunks": []}
            with open(self.resume_info_file, 'w') as f:
                json.dump(resume, f)
            self.log(f"[INFO] Created resume file: {self.resume_info_file}")

        resume_lock = threading.Lock()
        save_q = Queue()

        # — Сэйверы кадров и лейблов —
        def saver():
            while True:
                itm = save_q.get()
                if itm is None: break
                frame, img_p, lbl_p, lines, name = itm
                cv2.imwrite(img_p, frame)
                with open(lbl_p, 'w') as fw:
                    fw.write("\n".join(lines) + "\n")
                self.log(f"[INFO] Saved {name} ({len(lines)} labels)")
                save_q.task_done()

        savers = []
        for _ in range(self.save_workers):
            t = threading.Thread(target=saver, daemon=True)
            t.start()
            savers.append(t)

        # — Загружаем / создаём чанки —
        chunks = list(Path(self.chunks_folder).glob("*.ts"))
        if not chunks:
            self.log(f"[WARNING] Чанков нет → вызываю split_all()")
            self.split_all()
            chunks = list(Path(self.chunks_folder).glob("*.ts"))
            if not chunks:
                raise RuntimeError("Не удалось создать .ts-чанки!")

        # — Функция обработки одного .ts —
        def proc_chunk(ts_path: Path):
            base = ts_path.stem
            with resume_lock:
                if base in resume["processed_chunks"]:
                    self.log(f"[SKIP] {base}")
                    return

            cap = cv2.VideoCapture(str(ts_path))
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break

                # Центрированный crop 640×640
                h, w = frame.shape[:2]
                x0, y0 = (w-640)//2, (h-640)//2
                crop = frame[y0:y0+640, x0:x0+640]

                if idx % self.time_interval == 0:
                    results = self.model(crop, imgsz=(640,640), conf=self.detection_threshold)
                    lines = []
                    for r in results:
                        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                            cid = int(cls)
                            if cid in self.class_map.values():
                                x1,y1,x2,y2 = box
                                cx = ((x1+x2)/2)/640
                                cy = ((y1+y2)/2)/640
                                bw = (x2-x1)/640
                                bh = (y2-y1)/640
                                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                    if lines:
                        name = f"{base}_{idx:05d}.jpg"
                        img_p = os.path.join(self.images_folder, name)
                        lbl_p = os.path.join(self.labels_folder, name.replace(".jpg", ".txt"))
                        save_q.put((crop.copy(), img_p, lbl_p, lines, name))

                idx += 1

            cap.release()
            with resume_lock:
                resume["processed_chunks"].append(base)
                with open(self.resume_info_file, 'w') as f:
                    json.dump(resume, f)

        self.log(f"[INFO] Processing {len(chunks)} chunks with {self.process_workers} workers")
        with ThreadPoolExecutor(max_workers=self.process_workers) as ex:
            ex.map(proc_chunk, chunks)

        save_q.join()
        for _ in savers: save_q.put(None)
        for t in savers: t.join()

    def run(self):
        self.download_all()
        self.split_all()    # ← тут режем стримы на чанки
        self.process_all()
        self.log("[DONE]")

if __name__ == "__main__":
    StreamCut("configStreamCut.json").run()
