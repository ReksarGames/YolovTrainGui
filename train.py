import os
import torch
from ultralytics import YOLO

#  yolo export model=runs/train/auto/weights/best.pt format=onnx dynamic=true simplify=true

def train_yolo(
    model_path='models/yolov12s.pt',
    data_yaml='datasets/Valorant/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=None,
    project='runs/train',
    name='auto'
):
    # --- Определение устройства ---
    if device is None:
        if torch.cuda.is_available():
            device = 0
            print(f"[INFO] Найден GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("[WARNING] GPU не найден. Используется CPU.")

    # --- Проверка существования модели ---
    if not os.path.exists(model_path):
        print(f"[ERROR] Файл модели не найден: {model_path}")
        return

    # --- Загрузка модели ---
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить модель: {e}")
        return

    # --- Запуск обучения ---
    print(f"[INFO] Запуск обучения YOLOv12...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        amp=True,
        plots=True,
        save_json=True,

        # 🎮 Аугментации
        hsv_h=0.02,  # Hue shift — больше контраста по цвету
        hsv_s=0.7,  # Saturation shift
        hsv_v=0.5,  # Brightness shift
        fliplr=0.5,  # Горизонтальное отражение
        flipud=0.0,  # Вертикальное отключено
        scale=0.6,  # Изменение масштаба
        translate=0.2,  # Смещение объектов
        shear=0.1,  # Лёгкий наклон
        erasing=0.4,  # Random Erase помогает при повторяющихся текстурах
        mosaic=1.0,  # Комбинирует 4 изображения
        mixup=0.1,  # Комбинирование двух изображений
        copy_paste=0.05  # Вставка объектов в другие сцены
    )

    print(f"[INFO] ✅ Обучение завершено. Модель сохранена в: {project}/{name}/weights/best.pt")

if __name__ == '__main__':
    train_yolo()
