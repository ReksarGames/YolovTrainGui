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

        for p in (self.raw_stream_folder, self.chunks_folder, self.output_folder):
            os.makedirs(p, exist_ok=True)

        self.images_folder = os.path.join(self.output_folder, "images")
        self.labels_folder = os.path.join(self.output_folder, "labels")
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.labels_folder, exist_ok=True)

        # ——— Параметры ———
        selected_sources = cfg.get("selected_video_sources", [])
        use_selected_only = cfg.get("use_selected_only", False)
        self.video_sources = selected_sources if (use_selected_only and selected_sources) else cfg["video_sources"]
        self.chunks_per_stream   = cfg["chunks_per_stream"]
        self.time_interval       = cfg["time_interval"]
        self.detection_threshold = cfg["detection_threshold"]
        self.model_path          = cfg["model_path"]
        self.class_names = cfg.get("classes", None)
        self.twitch_cookies_path = cfg.get("twitch_cookies_path", None)

        # ——— Размер сохранения скринов ———
        self.crop_size = 900

        # ——— Параллельность ———
        self.max_download_workers = cfg["max_download_workers"]
        self.split_workers        = cfg["split_workers"]
        self.process_workers      = cfg["process_workers"]
        self.save_workers         = cfg["save_workers"]

        # ——— Архив скачанных и резюме ———
        self.download_archive = cfg["download_archive"]
        self.resume_info_file = cfg["resume_info_file"]
        self.ffmpeg_path      = cfg["ffmpeg_path"]
        self.download_quality = cfg.get("download_quality", "best")

        # ——— Модель ———
        self.model = YOLO(self.model_path)
        self.stop_flag = False
        self.on_download_info = None
        self.on_download_progress = None

    def stop(self):
        self.stop_flag = True

    def log(self, msg: str):
        print(msg)

    def download_all(self):
        """Параллельно скачиваем VOD в raw_stream_folder. Сначала без cookies, если не выйдет — пробуем с cookies."""
        if self.stop_flag:
            return

        import yt_dlp
        from yt_dlp.utils import DownloadCancelled

        # Проверяем файл cookies
        cookies_file = None
        if getattr(self, "twitch_cookies_path", None):
            if not os.path.isfile(self.twitch_cookies_path):
                raise FileNotFoundError(
                    f"[ERROR] Файл cookies не найден: {self.twitch_cookies_path}\n"
                    f"Проверь путь в конфиге."
                )
            if os.path.getsize(self.twitch_cookies_path) == 0:
                raise ValueError(
                    f"[ERROR] Файл cookies пустой: {self.twitch_cookies_path}\n"
                    f"Экспортируй cookies из браузера в формате Netscape."
                )

            with open(self.twitch_cookies_path, "rb") as f:
                header = f.read(16)
            if header.startswith(b"SQLite format 3"):
                self.log("[WARNING] Cookies file looks like a browser database (SQLite). Export Netscape cookies instead.")
            else:
                raw = None
                for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
                    try:
                        raw = Path(self.twitch_cookies_path).read_text(encoding=enc)
                        break
                    except Exception:
                        raw = None
                if raw is None:
                    self.log("[WARNING] Cookies file encoding is unknown. Cookies will be ignored.")
                else:
                    cookies_tmp = Path(self.output_folder).parent / "cookies_utf8.txt"
                    cookies_tmp.write_text(raw, encoding="utf-8")
                    cookies_file = str(cookies_tmp)
                    if "twitch.tv" not in raw:
                        self.log("[WARNING] Cookies file does not contain twitch.tv; auth might fail.")

        def build_opts(progress_hook):
            opts = {
                "outtmpl": os.path.join(self.raw_stream_folder, "%(id)s.%(ext)s"),
                "format": self._get_download_format(),
                "download_archive": self.download_archive,
                "progress_hooks": [progress_hook],
                "quiet": True,
                "no_warnings": True,
                "ignoreconfig": True,
            }
            if cookies_file:
                opts["cookiefile"] = cookies_file
            return opts

        def download_one(url):
            if self.stop_flag:
                return
            try:
                def progress_hook(d):
                    if self.stop_flag:
                        raise DownloadCancelled()
                    if d.get("status") == "downloading":
                        total = d.get("total_bytes") or d.get("total_bytes_estimate")
                        downloaded = d.get("downloaded_bytes")
                        percent = 0
                        if total and downloaded:
                            try:
                                percent = int((downloaded / total) * 100)
                            except Exception:
                                percent = 0
                        info = {
                            "title": d.get("info_dict", {}).get("title"),
                            "total_bytes": total,
                            "downloaded_bytes": downloaded,
                            "speed_bytes": d.get("speed"),
                            "eta": d.get("eta"),
                            "percent": percent,
                            "source_key": url,
                            "url": url,
                        }
                        if self.on_download_progress:
                            self.on_download_progress(info)

                with yt_dlp.YoutubeDL(build_opts(progress_hook)) as ydl:
                    info = ydl.extract_info(url, download=False)
                    title = info.get("title")
                    size = info.get("filesize") or info.get("filesize_approx")
                    if self.on_download_info:
                        self.on_download_info({"title": title, "size_bytes": size, "source_key": url, "url": url})
                    ydl.download([url])
                self.log(f"[DOWNLOAD] {url} (ok)")
            except DownloadCancelled:
                self.log("[INFO] Download cancelled by user.")
            except Exception as e:
                self.log(f"[ERROR] Download failed for {url}: {e}")

        if self.max_download_workers <= 1 or len(self.video_sources) <= 1:
            for url in self.video_sources:
                if self.stop_flag:
                    break
                download_one(url)
            return

        self.log(f"[DOWNLOAD] Using {self.max_download_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.max_download_workers) as ex:
            futures = [ex.submit(download_one, url) for url in self.video_sources]
            for fut in futures:
                if self.stop_flag:
                    break
                try:
                    fut.result()
                except Exception:
                    pass

    def _get_download_format(self):
        q = str(self.download_quality or "best").lower().strip()
        if q in ("best", "max", "highest"):
            return "best"
        if q in ("720", "720p"):
            return "best[height<=720]"
        if q in ("360", "360p"):
            return "best[height<=360]"
        return q

    def _get_duration(self, path: Path) -> float:
        """Возвращает длительность видео в секундах через ffprobe.exe."""
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

    def split_all(self):
        """
        Разбиваем скачанные видео на .ts-чанки.
        Каждый файл режется на self.chunks_per_stream равных сегментов,
        покрывающих всю его длину. Если уже нарезано достаточно чанков — пропускаем.
        """
        os.makedirs(self.chunks_folder, exist_ok=True)
        files = list(Path(self.raw_stream_folder).glob("*.*"))
        if not files:
            self.log(f"[SPLIT] ⚠ Нет исходных видео в {self.raw_stream_folder}")
            return
        if self.stop_flag:
            return

        self.log(f"[SPLIT] Режу {len(files)} файлов на {self.chunks_per_stream} частей…")

        def split_file(path: Path):
            if self.stop_flag:
                return
            base = path.stem
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
            if not self.stop_flag:
                ex.map(split_file, files)

    def process_all(self):
        """
        Обрабатывает .ts-чанки: выполняет инференс моделью YOLO,
        сохраняет изображения и метки в YOLO-формате.
        Поддерживает фильтрацию по именам классов и переиндексацию.
        """
        os.makedirs(os.path.dirname(self.resume_info_file), exist_ok=True)
        if self.stop_flag:
            return
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

        # ——— Классы из модели ———
        self.original_classes = self.model.model.names  # {0: 'person', 1: 'car', ...}

        # ——— Если задан class_names — фильтруем по имени, иначе сохраняем всё ———
        if hasattr(self, "class_names"):
            self.class_names = [str(n).strip().lower() for n in self.class_names]
            self.class_index_map = {
                orig_idx: new_idx
                for new_idx, (orig_idx, name) in enumerate([
                    (i, n) for i, n in self.original_classes.items()
                    if n.lower() in self.class_names
                ])
            }
            self.log(f"[INFO] Используем классы: {self.class_index_map}")
        elif hasattr(self, "class_map"):
            self.class_map = {int(k): int(v) for k, v in self.class_map.items()}
            self.class_index_map = {v: k for k, v in self.class_map.items()}
            self.log(f"[INFO] Используем class_map: {self.class_index_map}")
        else:
            self.class_index_map = None
            self.log(f"[INFO] Классы не фильтруются. Сохраняем все.")

        def saver():
            while True:
                if self.stop_flag:
                    break
                itm = save_q.get()
                if itm is None:
                    break
                frame, img_p, lbl_p, lines, name = itm
                cv2.imwrite(img_p, frame)
                with open(lbl_p, 'w') as fw:
                    fw.write("\n".join(lines) + "\n")
                self.log(f"[SAVE] {name} ({len(lines)} объектов)")
                save_q.task_done()

        savers = []
        for _ in range(self.save_workers):
            t = threading.Thread(target=saver, daemon=True)
            t.start()
            savers.append(t)

        chunks = list(Path(self.chunks_folder).glob("*.ts"))
        if not chunks:
            self.log(f"[WARNING] Чанков нет → вызываю split_all()")
            self.split_all()
            chunks = list(Path(self.chunks_folder).glob("*.ts"))
            if not chunks:
                raise RuntimeError("Не удалось создать .ts-чанки!")

        def proc_chunk(ts_path: Path):
            base = ts_path.stem
            with resume_lock:
                if base in resume["processed_chunks"]:
                    self.log(f"[SKIP] {base}")
                    return

            cap = cv2.VideoCapture(str(ts_path))
            idx = 0
            while True:
                if self.stop_flag:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                h, w = frame.shape[:2]
                crop_size = self.crop_size

                # если кадр меньше crop_size — уменьшаем crop_size до минимального измерения
                if crop_size > min(w, h):
                    crop_size = min(w, h)

                # координаты центра
                x0 = max((w - crop_size) // 2, 0)
                y0 = max((h - crop_size) // 2, 0)
                x1 = x0 + crop_size
                y1 = y0 + crop_size

                # фиксированное центральное окно
                crop = frame[y0:y1, x0:x1]

                if idx % self.time_interval == 0:
                    results = self.model(crop, imgsz=(640, 640), conf=self.detection_threshold)
                    lines = []
                    for r in results:
                        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                            original_cid = int(cls)

                            if self.class_index_map is None:
                                new_cid = original_cid
                            elif original_cid in self.class_index_map:
                                new_cid = self.class_index_map[original_cid]
                            else:
                                continue  # класс не в списке — пропустить

                            x1, y1, x2, y2 = box
                            cx = ((x1 + x2) / 2) / crop_size
                            cy = ((y1 + y2) / 2) / crop_size
                            bw = (x2 - x1) / crop_size
                            bh = (y2 - y1) / crop_size

                            lines.append(f"{new_cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                    if lines:
                        name = f"{base}_{idx:05d}.jpg"
                        img_p = os.path.join(self.images_folder, name)
                        lbl_p = os.path.join(self.labels_folder, name.replace(".jpg", ".txt"))
                        save_q.put((crop.copy(), img_p, lbl_p, lines, name))

                idx += 1

            cap.release()
            with resume_lock:
                if base not in resume["processed_chunks"]:
                    resume["processed_chunks"].append(base)
                    with open(self.resume_info_file, 'w') as f:
                        json.dump(resume, f)

        if self.stop_flag:
            return
        self.log(f"[INFO] Processing {len(chunks)} chunks with {self.process_workers} workers")
        with ThreadPoolExecutor(max_workers=self.process_workers) as ex:
            if not self.stop_flag:
                ex.map(proc_chunk, chunks)

        save_q.join()
        for _ in savers:
            save_q.put(None)
        for t in savers:
            t.join()

    def run(self):
        self.download_all()
        if self.stop_flag:
            return
        self.split_all()
        if self.stop_flag:
            return
        self.process_all()
        if not self.stop_flag:
            self.log("[DONE]")

if __name__ == "__main__":
    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "configStreamCut.json"
    StreamCut(str(default_cfg)).run()
