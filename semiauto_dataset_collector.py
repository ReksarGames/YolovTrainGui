import os
import time
import json
import logging
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import mss
from screeninfo import get_monitors

class ScreenCapture:
    """
    Класс для захвата экрана, детекции и сохранения данных.
    Всё «тяжёлое» работает внутри capture_and_display();
    его можно запустить напрямую через метод run().
    """
    def __init__(self, config, output_folder, log_callback=print):
        self.config = config
        self.output_folder = output_folder
        self.log = log_callback

        self.stop_flag = False
        self.last_detection_time = 0

        # Загружаем модель в FP16 на CUDA
        self.model = YOLO(self.config["model_path"], verbose=False).to('cuda').half()

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
        img_path = os.path.join(self.output_folder, 'images',  f'{ts}.jpg')
        lbl_path = os.path.join(self.output_folder, 'labels', f'{ts}.txt')

        cv2.imwrite(img_path, frame)
        with open(lbl_path, 'w') as f:
            for box in results[0].boxes:
                cls = int(box.cls)
                x_c, y_c, w, h = box.xywh[0]
                x_c /= frame.shape[1]
                y_c /= frame.shape[0]
                w   /= frame.shape[1]
                h   /= frame.shape[0]
                f.write(f"{cls} {x_c} {y_c} {w} {h}\n")

        self.log(f"[INFO] Saved image and label: {os.path.basename(img_path)}, {os.path.basename(lbl_path)}")

    def draw_boxes(self, frame, results):
        img = frame.copy()
        for box in results[0].boxes:
            cls = int(box.cls)
            if cls not in self.config["classes"]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            conf = box.conf.item()
            cv2.putText(img, f"{cls} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return img

    def capture_and_display(self):
        # Основной цикл: инициализация → захват → детекция → сохранение → отображение
        self.capture_init()
        self.create_folders()
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        while not self.stop_flag:
            frame = self.take_shot()
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb)

            if results and len(results[0].boxes) > 0:
                now = time.time()
                if now - self.last_detection_time > 1:
                    self.save_image_and_labels(frame, results)
                    self.last_detection_time = now

                    annotated = self.draw_boxes(frame, results)
                    cv2.imshow("Detection Result", annotated)
                    self.log(f"[INFO] Detections: {len(results[0].boxes)}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_flag = True

            time.sleep(0.01)

        cv2.destroyAllWindows()

    def run(self):
        """Публичный метод для запуска из другого кода."""
        self.capture_and_display()


def load_config(config_file="config.json"):
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    config = load_config()
    out = config["output_folder"]
    sc = ScreenCapture(config, out)
    sc.run()

if __name__ == '__main__':
    main()
