import os
import sys
import json
import torch
import argparse
from ultralytics import YOLO


def load_config(path='config.json'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def train_yolo(
    model_path='models/yolov12s.pt',
    data_yaml='datasets/Valorant/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=None,
    project='runs/train',
    name='auto',
    resume=False,
    log=print,
    stop_callback=lambda: False  # 👈 новый параметр
):
    try:
        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'
            log(f"[INFO] Используется устройство: {device}")

        if not resume and not os.path.exists(model_path):
            log(f"[ERROR] Файл модели не найден: {model_path}")
            return 1

        if not os.path.exists(data_yaml):
            log(f"[ERROR] Файл data.yaml не найден: {data_yaml}")
            return 1

        log(f"[INFO] Модель: {model_path}")
        log(f"[INFO] Data.yaml: {data_yaml}")
        log(f"[INFO] Проект: {project}/{name}")
        log(f"[INFO] Эпох: {epochs} | Размер: {imgsz} | Batch: {batch}")
        log(f"[INFO] Режим: {'--resume' if resume else 'новое обучение'}")

        if stop_callback():
            log("[INFO] Обучение остановлено до начала.")
            return 1

        model = YOLO(model_path if not resume else None)

        log("[INFO] Запуск обучения...")
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            resume=resume,
            amp=True,
            plots=True,
            save_json=True,
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.5,
            fliplr=0.5,
            flipud=0.0,
            scale=0.6,
            translate=0.2,
            shear=0.1,
            erasing=0.4,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.05
        )

        if stop_callback():
            log("[INFO] Обучение остановлено вручную.")
            return 1

        log(f"[INFO] ✅ Обучение завершено. Модель: {project}/{name}/weights/best.pt")
        return 0

    except Exception as e:
        log(f"[ERROR] Исключение: {str(e)}")
        return 1


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='models/yolov12s.pt')
        parser.add_argument('--data')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--imgsz', type=int, default=640)
        parser.add_argument('--batch', type=int, default=16)
        parser.add_argument('--device', default=None)
        parser.add_argument('--project', default='runs/train')
        parser.add_argument('--name', default='auto')
        parser.add_argument('--resume', action='store_true')
        args = parser.parse_args()

        if not args.data:
            cfg = load_config()
            args.data = cfg.get("last_data_yaml")
        if not args.data:
            parser.error("--data обязательно")

        return args

    args = parse_args()
    sys.exit(train_yolo(
        model_path=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        log=print
    ))
