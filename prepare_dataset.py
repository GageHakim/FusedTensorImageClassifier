
import os
import random
import shutil

def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
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

            files = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
            random.shuffle(files)

            train_split = int(len(files) * split_ratio[0])
            val_split = int(len(files) * (split_ratio[0] + split_ratio[1]))

            train_files = files[:train_split]
            val_files = files[train_split:val_split]
            test_files = files[val_split:]

            for f in train_files:
                shutil.move(os.path.join(class_source_path, f), os.path.join(train_class_dir, f))
            for f in val_files:
                shutil.move(os.path.join(class_source_path, f), os.path.join(val_class_dir, f))
            for f in test_files:
                shutil.move(os.path.join(class_source_path, f), os.path.join(test_class_dir, f))

if __name__ == '__main__':
    source_directory = 'dataset/bacteria_resized_224'
    train_directory = 'dataset/train'
    val_directory = 'dataset/val'
    test_directory = 'dataset/test'
    split_data(source_directory, train_directory, val_directory, test_directory)
