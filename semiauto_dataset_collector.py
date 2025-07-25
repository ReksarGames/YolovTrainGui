import time
import cv2
import numpy as np
import os
import json
from datetime import datetime
from ultralytics import YOLO
import mss  # Используем только mss для захвата экрана
from screeninfo import get_monitors  # Для получения размера экрана
import logging


class ScreenCapture:
    def __init__(self, config, output_folder):
        self.config = config
        self.output_folder = output_folder
        self.stop_flag = False
        self.saved_frame_count = 0
        self.region = None
        self.camera = None
        self.target_classes = config["classes"]  # Извлекаем целевые классы из конфига
        self.last_detection_time = 0  # Время последней детекции для контроля вывода
        self.last_save_time = 0  # Время последнего сохранения

        # Загружаем модель с оптимизацией для производительности (например, FP16 для повышения скорости)
        self.model = YOLO(self.config["model_path"], verbose=False).to('cuda').half()  # Используем FP16 для ускорения

    def create_folders(self):
        # Проверяем, существуют ли папки, и если нет, создаем их
        os.makedirs(os.path.join(self.output_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, 'labels'), exist_ok=True)

    def capture_init(self):
        # Получаем текущий размер экрана
        monitor = get_monitors()[0]  # Получаем первый монитор
        screen_width = monitor.width
        screen_height = monitor.height

        # Задаем размер области захвата
        crop_width = self.config["grabber"]["width"]
        crop_height = self.config["grabber"]["height"]

        # Центральная область экрана для захвата
        x = (screen_width - crop_width) // 2
        y = (screen_height - crop_height) // 2
        self.region = {'top': y, 'left': x, 'width': crop_width, 'height': crop_height}
        self.camera = mss.mss()

        # Выводим информацию о размере экрана
        print(f"[INFO] Screen resolution: {screen_width}x{screen_height}")
        print(f"[INFO] Capturing region: {self.region}")

    def take_shot(self):
        # Захват изображения с помощью mss
        img = self.camera.grab(self.region)
        img = np.array(img)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def save_image_and_labels(self, frame, results, output_folder):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        img_name = f'{timestamp}.jpg'
        label_name = f'{timestamp}.txt'

        # Сохраняем изображение
        img_bgr = frame  # Сохраняем в оригинальных цветах (BGR)
        img_path = os.path.join(output_folder, 'images', img_name)
        cv2.imwrite(img_path, img_bgr)

        # Сохраняем метки в формате YOLO
        label_path = os.path.join(output_folder, 'labels', label_name)
        with open(label_path, 'w') as f:
            for box in results[0].boxes:
                cls = int(box.cls)  # Класс объекта
                x_center, y_center, width, height = box.xywh[0]  # Получаем координаты и размер бокса

                # Нормализация координат относительно размеров изображения
                x_center /= frame.shape[1]
                y_center /= frame.shape[0]
                width /= frame.shape[1]
                height /= frame.shape[0]

                # Записываем метку в файл
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

        print(f"[INFO] Saved image and label: {img_name}, {label_name}")  # Общая информация

    def draw_boxes(self, frame, results):
        # Рисуем боксы на изображении
        annotated_img = frame.copy()
        for box in results[0].boxes:
            class_id = int(box.cls)
            # Пропускаем классы, которые не являются целевыми
            if class_id not in self.target_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Используем .item() для извлечения числа из тензора
            label = f"{class_id} {box.conf.item():.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated_img

    def capture_and_display(self):
        self.capture_init()
        self.create_folders()

        # Отключаем логирование, чтобы избежать лишнего вывода
        logging.getLogger("ultralytics").setLevel(logging.ERROR)  # Отключаем логирование от YOLO

        while not self.stop_flag:
            # Захват кадра с задержкой
            frame = self.take_shot()

            # Анализ изображения с помощью модели (с задержкой)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(img_rgb)

            # Если есть детекции
            if len(results[0].boxes) > 0:
                current_time = time.time()

                # Добавляем контроль времени для вывода в консоль (раз в 1 секунду)
                if current_time - self.last_detection_time > 1:  # 1 секунда между выводами
                    # Сохраняем изображение и метки
                    self.save_image_and_labels(frame, results, self.output_folder)
                    self.last_detection_time = current_time  # Обновляем время последней детекции

                    # Рисуем боксы на изображении
                    annotated_img = self.draw_boxes(frame, results)

                    # Отображаем изображение с аннотированными боксами
                    cv2.imshow("Detection Result", annotated_img)

                    # Логируем только если есть детекции
                    print(f"[INFO] Detections found: {len(results[0].boxes)}")

            # Добавляем небольшую задержку, чтобы избежать перегрузки процессора
            time.sleep(0.01)

            # Выход по нажатию клавиши 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_flag = True
                break


def load_config(config_file="config.json"):
    """Загрузка конфигурационного файла."""
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    config = load_config("config.json")

    output_folder = config["output_folder"]  # Папка для вывода данных
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels'), exist_ok=True)

    # Создаем экземпляр ScreenCapture
    screen_capture = ScreenCapture(config, output_folder)
    screen_capture.capture_and_display()


if __name__ == '__main__':
    main()
