import os
import shutil
import random
import json


# Функция для создания необходимых директорий в папке split
def create_dirs(output_base_dir):
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, split, 'labels'), exist_ok=True)


# Функция для разбиения датасета
def split_dataset(images_dir, labels_dir, output_base_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)

    # Рассчитываем количество файлов для train, val и test
    train_count = int(len(image_files) * train_ratio)
    val_count = int(len(image_files) * val_ratio)
    test_count = len(image_files) - train_count - val_count  # Остальные идут в тест

    # Разделяем файлы на train, val и test
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    # Разделяем файлы в соответствующие папки
    for split, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for file in files:
            image_path = os.path.join(images_dir, file)
            label_path = os.path.join(labels_dir, file.replace('.jpg', '.txt'))

            # Проверяем, что метка для изображения существует
            if not os.path.exists(label_path):
                print(f"Warning: Label file {label_path} for image {file} does not exist. Skipping this file.")
                continue  # Пропускаем, если метка не найдена

            output_image_path = os.path.join(output_base_dir, split, 'images', file)
            output_label_path = os.path.join(output_base_dir, split, 'labels', file.replace('.jpg', '.txt'))

            # Копируем изображения и метки
            shutil.copy(image_path, output_image_path)
            shutil.copy(label_path, output_label_path)

    print(f"Dataset split into train ({len(train_files)}), val ({len(val_files)}), and test ({len(test_files)}) sets.")


# Загрузка конфигурации из файла
def load_config(config_file="config.json"):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


# Функция для выполнения действия Split
def perform_split(config):
    # Параметры из конфигурации
    images_dir = os.path.join(config["data_folder"], "augmented_images")
    labels_dir = os.path.join(config["data_folder"], "augmented_labels")

    # Папка для вывода результата
    split_output_dir = os.path.join(config["data_folder"], "split_dataset")

    # Создаем нужные директории, если их нет
    create_dirs(split_output_dir)

    # Разделяем данные на train, val, test
    split_dataset(images_dir, labels_dir, split_output_dir)


# Главная функция
if __name__ == "__main__":
    config = load_config("config.json")  # Загружаем конфигурацию из файла

    # Выполняем split
    perform_split(config)
