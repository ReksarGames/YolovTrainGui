import os
import time
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import mss
from screeninfo import get_monitors

class ScreenCapture:
    def __init__(self, config, output_folder, log_callback=print, show_window=True):
        self.config = config
        self.output_folder = output_folder
        self.log = log_callback
        self.on_frame_ready = None
        self.stop_flag = False
        self.last_detection_time = 0
        self.show_window = show_window

        # Загрузка модели
        self.model = YOLO(self.config["model_path"], verbose=False).to('cuda').half()

        # Подготовка разрешённых классов и их переиндексации
        class_name_to_index = self.config["class_map"]
        self.class_remap = {}
        for i, name in enumerate(self.config["classes"]):
            model_class_idx = class_name_to_index.get(name)
            if model_class_idx is not None and model_class_idx != 99:
                self.class_remap[model_class_idx] = i  # YOLO class → dataset class

        self.allowed_class_ids = set(self.class_remap.keys())

    def create_folders(self):
        os.makedirs(os.path.join(self.output_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, 'labels'), exist_ok=True)

    def capture_init(self):
        mon = get_monitors()[0]
        sw, sh = mon.width, mon.height
        cw = self.config["grabber"]["width"]
        ch = self.config["grabber"]["height"]
        x = (sw - cw) // 2
        y = (sh - ch) // 2
        self.region = {'top': y, 'left': x, 'width': cw, 'height': ch}
        self.camera = mss.mss()
        self.log(f"[INFO] Screen resolution: {sw}x{sh}")
        self.log(f"[INFO] Capturing region: {self.region}")

    def take_shot(self):
        img = self.camera.grab(self.region)
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def save_image_and_labels(self, frame, results):
        ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
        img_path = os.path.join(self.output_folder, 'images', f'{ts}.jpg')
        lbl_path = os.path.join(self.output_folder, 'labels', f'{ts}.txt')

        height, width = frame.shape[:2]
        lines = []

        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            if cls not in self.allowed_class_ids:
                self.log(f"[DEBUG] Class ID {cls} skipped")
                continue
            if conf < self.config.get("detection_threshold", 0.5):
                continue

            mapped_cls = self.class_remap[cls]
            x_c, y_c, w, h = box.xywh[0].cpu().numpy()
            x_c /= width
            y_c /= height
            w /= width
            h /= height
            lines.append(f"{mapped_cls} {x_c} {y_c} {w} {h}")

        # Сохраняем только если есть метки
        if lines:
            cv2.imwrite(img_path, frame)
            with open(lbl_path, 'w') as f:
                f.write("\n".join(lines) + "\n")
            self.log(f"[INFO] Saved image and {len(lines)} labels → {os.path.basename(img_path)}")
        else:
            self.log(f"[WARNING] No valid detections to save → skipping image")

    def draw_boxes(self, frame, results):
        img = frame.copy()
        for box in results[0].boxes:
            cls = int(box.cls)
            if cls not in self.allowed_class_ids:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            mapped_cls = self.class_remap.get(cls, cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{mapped_cls} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    def capture_and_display(self):
        self.capture_init()
        self.create_folders()
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        while not self.stop_flag:
            frame = self.take_shot()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb)

            annotated = frame
            if results and len(results[0].boxes) > 0:
                now = time.time()
                if now - self.last_detection_time > self.config.get("save_interval", 1):
                    self.save_image_and_labels(frame, results)
                    self.last_detection_time = now

                annotated = self.draw_boxes(frame, results)
                if self.show_window:
                    cv2.imshow("Detection Result", annotated)
                # self.log(f"[INFO] Detections: {len(results[0].boxes)}")

            if self.on_frame_ready:
                try:
                    self.on_frame_ready(annotated)
                except Exception:
                    pass

            if self.show_window:
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    self.stop_flag = True

            time.sleep(0.01)

        if self.show_window:
            cv2.destroyAllWindows()

    def run(self):
        self.capture_and_display()


def load_config(config_file=None):
    if config_file is None:
        config_file = Path(__file__).resolve().parent.parent / "configs" / "config.json"
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    config = load_config()
    out = config["output_folder"]
    sc = ScreenCapture(config, out)
    sc.run()

if __name__ == '__main__':
    main()
