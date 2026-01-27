import os
import sys
import json
from pathlib import Path
import torch
import argparse
from ultralytics import YOLO


def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "config.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def train_yolo(
    model_path='models/sunxds_0.7.6.pt',
    data_yaml='datasets/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=None,
    project='runs/train',
    name='auto3',
    resume=False,
    exist_ok=True,
    save_period=100,
    log=print,
    stop_callback=lambda: False,
    patience = 20,
    overrides=None
):
    try:
        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'
            log(f"[INFO] Using device: {device}")

        if not resume and not os.path.exists(model_path):
            log(f"[ERROR] Model file not found: {model_path}")
            return 1

        if not os.path.exists(data_yaml):
            log(f"[ERROR] data.yaml file not found: {data_yaml}")
            return 1

        log(f"[INFO] Model: {model_path}")
        log(f"[INFO] Data.yaml: {data_yaml}")
        log(f"[INFO] Project: {project}/{name}")
        log(f"[INFO] Epochs: {epochs} | ImgSize: {imgsz} | Batch: {batch}")
        log(f"[INFO] Mode: {'--resume' if resume else 'new training'}")

        if stop_callback():
            log("[INFO] Training was interrupted before start.")
            return 1

        model = YOLO(model_path)

        log("[INFO] Starting training...")
        train_kwargs = dict(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            patience=patience,
            amp=True,
            plots=True,
            save_json=True,
            save_period=save_period,
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.5,
            fliplr=0.5,
            flipud=0.0,
            scale=0.6,
            translate=0.2,
            shear=0.1,
            erasing=0.3,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.05,
            optimizer='SGD',
            lr0=0.001
        )

        if overrides and isinstance(overrides, dict):
            train_kwargs.update(overrides)

        model.train(**train_kwargs)

        if stop_callback():
            log("[INFO] Training was manually interrupted.")
            return 1

        log(f"[INFO] ✅ Training completed. Model saved at: {project}/{name}/weights/best.pt")
        return 0

    except Exception as e:
        log(f"[ERROR] Exception during training: {str(e)}")
        return 1


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='models/sunxds_0.7.6.pt')
        parser.add_argument('--data')
        parser.add_argument('--epochs', type=int, default=160)
        parser.add_argument('--imgsz', type=int, default=640)
        parser.add_argument('--batch', type=int, default=16)
        parser.add_argument('--device', default=None)
        parser.add_argument('--project', default='runs/train1000')
        parser.add_argument('--name', default='auto2')
        parser.add_argument('--resume', action='store_true')
        args = parser.parse_args()

        if not args.data:
            cfg = load_config()
            args.data = cfg.get("last_data_yaml")
        if not args.data:
            parser.error("--data is required")

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
