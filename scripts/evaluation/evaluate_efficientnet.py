import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from efficientnet_pytorch import EfficientNet
import argparse
import torch.nn.functional as F


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Generates and saves a confusion matrix heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.close()


def get_test_loader(data_dir, batch_size):
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader, test_dataset.classes


def evaluate_efficientnet(model_path='best_model.pth', data_dir='dataset', batch_size=32):
    """
    Evaluates the EfficientNet model and returns a dictionary of metrics.
    """
    # --- Check for model file ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            f"Please train the model first by running `python3 EfficientNet-PyTorch/train.py`"
        )
        
    # --- Hardware Optimization ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # --- Data ---
    test_loader, class_names = get_test_loader(data_dir, batch_size)
    num_classes = len(class_names)

    # --- Load Model ---
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Evaluation Loop ---
    all_preds = []
    all_labels = []
    total_time = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            if device.type in ['cuda', 'mps']:
                if device.type == 'mps':
                    torch.mps.synchronize()
                else:
                    torch.cuda.synchronize()
            start_time = time.time()

            outputs = model(images)

            if device.type in ['cuda', 'mps']:
                if device.type == 'mps':
                    torch.mps.synchronize()
                else:
                    torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics ---
    num_images = len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    avg_total_time_ms = (total_time / num_images) * 1000 if num_images > 0 else 0
    fps_total = 1.0 / (total_time / num_images) if total_time > 0 else 0

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_path='efficientnet_confusion_matrix.png')

    return {
        'accuracy': accuracy,
        'avg_extraction_time_ms': 0,  # Not applicable
        'avg_classification_time_ms': avg_total_time_ms, # Counts as classification
        'avg_total_time_ms': avg_total_time_ms,
        'fps_extraction': float('inf'), # Not applicable
        'fps_classification': fps_total,
        'fps_total': fps_total,
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate an EfficientNet model.')
    parser.add_argument('--model-path', type=str, default='best_model.pth',
                        help='Path to the trained model weights.')
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='Path to the dataset directory.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation.')
    args = parser.parse_args()

    metrics = evaluate_efficientnet(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    print("-" * 30)
    print(f"Global Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Inference Speed: {metrics['avg_total_time_ms']:.2f} ms/image ({metrics['fps_total']:.1f} FPS)")
    print("-" * 30)

    # The detailed report is not part of the returned metrics, so we'd need to re-calculate or adjust `evaluate_efficientnet` if needed.
    # For now, we just print the main metrics.
    # To get the classification report, we would need the predictions and labels from the function.


if __name__ == '__main__':
    main()
