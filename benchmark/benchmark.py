#!/usr/bin/env python3
"""
Benchmark script for YOLO models.
Universal postprocess - all models use RGB 0-1.
"""
import cv2
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import onnxruntime as rt
from glob import glob
from infer_function import read_img, draw_boxes_v8, postprocess

CONF_THRES = 0.3
IOU_THRES = 0.1
TEST_IMAGE = 'apex-legends-season-19-meta-firefight.jpg'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def _collect_images(images_dir):
    if not images_dir or not os.path.isdir(images_dir):
        return []
    paths = []
    for name in os.listdir(images_dir):
        path = os.path.join(images_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            paths.append(path)
    return sorted(paths)


def run_benchmark(models_dir=None, images_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if images_dir:
        images = _collect_images(images_dir)
        if not images:
            print(f"ERROR: No images found in {images_dir}")
            return
    else:
        images = [os.path.join(script_dir, TEST_IMAGE)]

    if models_dir:
        models = glob(os.path.join(models_dir, '**', '*.onnx'), recursive=True)
    else:
        weights_dir = os.path.join(script_dir, '..', '..')
        models = glob(os.path.join(weights_dir, '**', '*.onnx'), recursive=True)

    multi_image = len(images) > 1

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"ERROR: Cannot load {img_path}")
            continue

        h, w = img.shape[:2]
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        img_cropped = img[start_y:start_y+crop_size, start_x:start_x+crop_size]

        print(f"Image: {os.path.basename(img_path)} ({w}x{h} -> crop {crop_size}x{crop_size})")
        print(f"{'Model':<45} {'Shape':<12} {'nCls':<5} {'#':<3} {'Classes':<12} {'Conf1':<6}")
        print("-" * 90)

        for model_path in sorted(models):
            name = os.path.basename(model_path)

            # Skip non-converted legacy models
            if '_converted' not in name and '_modern' not in name:
                # Check if converted version exists
                base, ext = os.path.splitext(model_path)
                if os.path.exists(f"{base}_modern{ext}"):
                    continue

            try:
                session = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                input_info = session.get_inputs()[0]
                input_shape = input_info.shape
                metadata = session.get_modelmeta().custom_metadata_map

                if isinstance(input_shape[2], str):
                    imgsz = json.loads(metadata.get('imgsz', '[640, 640]'))
                    input_size = (imgsz[1], imgsz[0])
                else:
                    input_size = (input_shape[3], input_shape[2])

                blob = read_img(img_cropped, input_size)
                outputs = session.run(None, {input_info.name: blob})
                pred = outputs[0]
                pred_sq = np.squeeze(pred)

                shape_str = f"[{pred_sq.shape[0]},{pred_sq.shape[1]}]" if pred_sq.ndim == 2 else str(pred_sq.shape)

                rows, cols = pred_sq.shape
                if cols == 6:
                    num_classes = 1
                else:
                    num_classes = max(1, rows - 4)

                boxes, scores, classes = postprocess(pred, CONF_THRES, IOU_THRES, num_classes)

                if len(scores) > 0:
                    order = np.argsort(scores)[::-1]
                    boxes, scores, classes = boxes[order], scores[order], classes[order]

                cnt = len(boxes)
                conf1 = f"{scores[0]:.2f}" if cnt > 0 else "-"
                unique_cls = sorted(set(classes.astype(int))) if cnt > 0 else []
                cls_str = ",".join(map(str, unique_cls)) if unique_cls else "-"

                print(f"{name:<45} {shape_str:<12} {num_classes:<5} {cnt:<3} {cls_str:<12} {conf1:<6}")

                if cnt > 0:
                    img_resized = cv2.resize(img_cropped, input_size)
                    result_img = draw_boxes_v8(img_resized, boxes, scores, classes)
                    model_base = os.path.splitext(name)[0]
                    if multi_image:
                        img_base = os.path.splitext(os.path.basename(img_path))[0]
                        out_name = f"{img_base}__{model_base}.jpg"
                    else:
                        out_name = f"{model_base}.jpg"
                    cv2.imwrite(os.path.join(script_dir, out_name), result_img)

            except Exception as e:
                import traceback
                print(f"{name:<40} ERR: {str(e)[:35]}")
                # traceback.print_exc()

        print("")

    print(f"\nResults: {script_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', '-m', help='Models directory')
    parser.add_argument('--images-dir', '-i', help='Directory with images')
    args = parser.parse_args()
    run_benchmark(args.models, args.images_dir)
