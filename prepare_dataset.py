import os
import json
import random
import shutil
from PIL import Image


def process_and_split_data(source_dir, processed_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15), target_size=224):
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.json'):
                    json_path = os.path.join(subdir_path, filename)
                    image_path = os.path.join(
                        subdir_path, filename.replace('.json', '.jpg'))

                    if not os.path.exists(image_path):
                        continue

                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    if "labels" not in data or not data["labels"]:
                        continue

                    original_image = Image.open(image_path)
                    for i, label in enumerate(data['labels']):
                        class_name = label['class']
                        x, y, w, h = label['x'], label['y'], label['width'], label['height']

                        cropped_image = original_image.crop(
                            (x, y, x + w, y + h))

                        # Resize with padding
                        aspect_ratio = cropped_image.width / cropped_image.height
                        if cropped_image.width > cropped_image.height:
                            new_width = target_size
                            new_height = int(target_size / aspect_ratio)
                        else:
                            new_height = target_size
                            new_width = int(target_size * aspect_ratio)

                        resized_image = cropped_image.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS)

                        padded_image = Image.new(
                            "RGB", (target_size, target_size), (0, 0, 0))
                        paste_x = (target_size - new_width) // 2
                        paste_y = (target_size - new_height) // 2
                        padded_image.paste(resized_image, (paste_x, paste_y))

                        class_dir = os.path.join(processed_dir, class_name)
                        if not os.path.exists(class_dir):
                            os.makedirs(class_dir)

                        padded_image.save(os.path.join(
                            class_dir, f"{data['sample_id']}_{i}.jpg"))

    # Split the processed data
    split_data(processed_dir, train_dir, val_dir, test_dir, split_ratio)


def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for class_name in os.listdir(source_dir):
        class_source_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_source_path):
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)

            if not os.path.exists(train_class_dir):
                os.makedirs(train_class_dir)
            if not os.path.exists(val_class_dir):
                os.makedirs(val_class_dir)
            if not os.path.exists(test_class_dir):
                os.makedirs(test_class_dir)

            files = [f for f in os.listdir(class_source_path) if os.path.isfile(
                os.path.join(class_source_path, f))]
            random.shuffle(files)

            train_split = int(len(files) * split_ratio[0])
            val_split = int(len(files) * (split_ratio[0] + split_ratio[1]))

            train_files = files[:train_split]
            val_files = files[train_split:val_split]
            test_files = files[val_split:]

            for f in train_files:
                shutil.copy(os.path.join(class_source_path, f),
                            os.path.join(train_class_dir, f))
            for f in val_files:
                shutil.copy(os.path.join(class_source_path, f),
                            os.path.join(val_class_dir, f))
            for f in test_files:
                shutil.copy(os.path.join(class_source_path, f),
                            os.path.join(test_class_dir, f))


if __name__ == '__main__':
    source_directory = 'AGAR_dataset/lkj'
    processed_directory = 'dataset2/processed'
    train_directory = 'dataset2/train'
    val_directory = 'dataset2/val'
    test_directory = 'dataset2/test'
    process_and_split_data(source_directory, processed_directory,
                           train_directory, val_directory, test_directory)
