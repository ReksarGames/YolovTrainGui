import cv2
import os
import json
import numpy as np
from pathlib import Path

ref_point = []
cropping = False
current_labels = []
image_copy = None
class_names = {}
current_class_id = 0
jump_mode = False
jump_buffer = ""

class_colors = {}

class ColorGenerator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.colors = self.generate_colors()

    def generate_colors(self):
        colors = []

        # Assign fixed colors to the first two classes
        colors.append((0, 255, 0))  # Green for class 0
        colors.append((0, 0, 255))  # Red for class 1

        # Generate random colors for the rest
        for i in range(2, self.num_classes):
            color = tuple(np.random.randint(0, 256, 3).tolist())
            colors.append(color)
        return colors

    def get_color(self, class_id):
        return self.colors[class_id] if 0 <= class_id < self.num_classes else (255, 255, 255)  # Белый цвет по умолчанию для несуществующих классов


def load_config(config_file=None):
    if config_file is None:
        config_file = Path(__file__).resolve().parent.parent / "configs" / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    global class_names, current_class_id
    class_names = {i: name for i, name in enumerate(config["classes"])}
    current_class_id = 0

    return config


def generate_class_colors():
    global color_generator, class_colors
    color_generator = ColorGenerator(len(class_names))
    print("[INFO] Class colors:")
    for i, name in class_names.items():
        color = color_generator.get_color(i)
        class_colors[i] = color
        print(f"Class {i} ({name}): {color}")


def draw_labels_on_image(image_path, label_path):
    global current_labels

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return None, []

    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        return image, []

    with open(label_path, 'r') as file:
        lines = file.readlines()

    current_labels = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, width, height = map(float, parts)
        current_labels.append((class_id, x_center, y_center, width, height))

        img_h, img_w = image.shape[:2]
        x_center = int(x_center * img_w)
        y_center = int(y_center * img_h)
        width = int(width * img_w)
        height = int(height * img_h)

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        cv2.circle(image, (x_center, y_center), 5, (0, 0, 255), -1)

        class_color = class_colors.get(int(class_id), (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), class_color, 2)

        # ✨ Добавляем текст над боксом
        label_text = f"{int(class_id)}: {class_names.get(int(class_id), 'Unknown')}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label_text, (x1, max(y1 - 10, 20)), font, 0.6, class_color, 2, cv2.LINE_AA)

    return image, current_labels


def display_current_class(image):
    global current_class_id
    if not class_names:
        return
    text = f"Current class: {class_names[current_class_id]}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


def update_image_window(image, counter_text=None):
    if image is None:
        return
    view = image.copy()
    display_current_class(view)
    draw_hotkeys(view)
    if counter_text:
        draw_counter(view, counter_text)
    if jump_mode:
        draw_jump_prompt(view)
    cv2.imshow('Image with Labels', view)
    cv2.waitKey(1)


def window_closed():
    try:
        return cv2.getWindowProperty('Image with Labels', cv2.WND_PROP_VISIBLE) < 1
    except Exception:
        return False


def draw_hotkeys(image):
    h, w = image.shape[:2]
    lines = [
        "Hotkeys: D=save+next  A=save+prev  W/S=class  H=delete  Q=quit",
        "Mouse: RMB drag=add box  LMB drag=remove box",
        "Jump: J=go to index (type digits, Enter)",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    padding = 6
    line_height = 18
    box_height = padding * 2 + line_height * len(lines)
    y0 = h - box_height
    if y0 < 0:
        y0 = 0
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    y = y0 + padding + line_height - 4
    for line in lines:
        cv2.putText(image, line, (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height


def draw_counter(image, text):
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = w - tw - 10
    y = h - 8
    cv2.rectangle(image, (x - 6, y - th - 6), (w - 4, y + 6), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_jump_prompt(image):
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    prompt = f"Go to: {jump_buffer}"
    (tw, th), _ = cv2.getTextSize(prompt, font, font_scale, thickness)
    x = 10
    y = 60
    cv2.rectangle(image, (x - 6, y - th - 6), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(image, prompt, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image_copy

    if event == cv2.EVENT_RBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            temp_image = image_copy.copy()
            cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
            update_image_window(temp_image)

    elif event == cv2.EVENT_RBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        cv2.rectangle(image_copy, ref_point[0], ref_point[1], (0, 255, 0), 2)
        update_image_window(image_copy)

        add_label_from_selected_area(ref_point)

    elif event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        remove_labels_within_selected_area(ref_point)


def remove_labels_within_selected_area(ref_point):
    global current_labels, image_copy

    x1, y1 = ref_point[0]
    x2, y2 = ref_point[1]

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    img_h, img_w = image_copy.shape[:2]

    new_labels = []
    for label in current_labels:
        class_id, x_center, y_center, width, height = label
        x_center_pixel = int(x_center * img_w)
        y_center_pixel = int(y_center * img_h)

        if not (x1 <= x_center_pixel <= x2 and y1 <= y_center_pixel <= y2):
            new_labels.append(label)
        else:
            print(f"Label removed: {label}")

    if len(new_labels) == len(current_labels):
        print("No labels removed.")
    else:
        print(f"Labels after removal: {len(new_labels)}")

    current_labels = new_labels


def add_label_from_selected_area(ref_point):
    global current_labels, image_copy, current_class_id

    img_h, img_w = image_copy.shape[:2]
    x1, y1 = ref_point[0]
    x2, y2 = ref_point[1]

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = abs(x2 - x1) / img_w
    height = abs(y2 - y1) / img_h

    new_label = (current_class_id, x_center, y_center, width, height)
    current_labels.append(new_label)

    print(f"Added label: {new_label}")

    update_image_window(image_copy)


def save_labels(label_path):
    global current_labels

    with open(label_path, 'w') as file:
        for label in current_labels:
            file.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


def browse_images(data_folder):
    global image_copy, current_class_id, jump_mode, jump_buffer

    image_folder = os.path.join(data_folder, 'images')
    label_folder = os.path.join(data_folder, 'labels')

    def list_images():
        if not os.path.isdir(image_folder):
            return []
        files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        files.sort()
        return files

    image_files = list_images()
    current_image_index = 0

    while True:
        if not image_files:
            print("No images found.")
            break

        current_image_index = max(0, min(current_image_index, len(image_files) - 1))
        image_name = image_files[current_image_index]
        image_path = os.path.join(image_folder, image_name)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(label_folder, label_name)

        image, labels = draw_labels_on_image(image_path, label_path)
        if image is None:
            image_files = list_images()
            continue
        image_copy = image.copy()

        if image is not None:
            update_image_window(image_copy, f"{current_image_index + 1}/{len(image_files)}")
            cv2.setMouseCallback("Image with Labels", click_and_crop)

        print(f"Current Class: {class_names[current_class_id]}")

        key = -1
        while True:
            if window_closed():
                cv2.destroyAllWindows()
                return
            key = cv2.waitKey(30)
            if key != -1:
                break

        if jump_mode:
            if key == 27:  # ESC
                jump_mode = False
                jump_buffer = ""
            elif key in (8, 127):  # backspace
                jump_buffer = jump_buffer[:-1]
            elif key in (13, 10):  # enter
                if jump_buffer.isdigit():
                    target = int(jump_buffer) - 1
                    if 0 <= target < len(image_files):
                        current_image_index = target
                    else:
                        print("Index out of range.")
                else:
                    print("Invalid index.")
                jump_mode = False
                jump_buffer = ""
            elif 48 <= key <= 57:  # digits
                jump_buffer += chr(key)
            update_image_window(image_copy, f"{current_image_index + 1}/{len(image_files)}")
            continue

        if key == ord('d'):
            # Save changes and move to the next image
            save_labels(label_path)
            current_image_index += 1
        elif key == ord('a'):
            # Save changes and go to the previous image
            save_labels(label_path)
            current_image_index -= 1
        elif key == ord('h'):
            # Delete current image and labels file
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                if os.path.exists(label_path):
                    os.remove(label_path)
            except Exception as e:
                print(f"Delete failed: {e}")
            print(f"Deleted: {image_path} and {label_path}")
            image_files = list_images()
            if current_image_index >= len(image_files):
                current_image_index = len(image_files) - 1
            update_image_window(image_copy, f"{current_image_index + 1}/{len(image_files)}")
        elif key == ord('j'):
            jump_mode = True
            jump_buffer = ""
            update_image_window(image_copy, f"{current_image_index + 1}/{len(image_files)}")
        elif key == ord('q'):
            break
        elif key == ord('w'):
            # Switching to the next class
            switch_class(direction=1)
            update_image_window(image_copy, f"{current_image_index + 1}/{len(image_files)}")
        elif key == ord('s'):
            # Switch to previous class
            switch_class(direction=-1)
            update_image_window(image_copy, f"{current_image_index + 1}/{len(image_files)}")

    cv2.destroyAllWindows()


def switch_class(direction=1):
    global current_class_id
    current_class_id = (current_class_id + direction) % len(class_names)
    print(f"Switched to class: {class_names[current_class_id]}")
    update_image_window(image_copy)


config = load_config()

generate_class_colors()

label_folder = config.get("label_data_folder") or config.get("data_folder")
browse_images(label_folder)  # Folder containing "images" and "labels"
