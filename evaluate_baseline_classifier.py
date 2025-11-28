from model import Model as HiFiCModel
from default_config import ModelModes, hific_args
from helpers import utils
from baseline_classifier import BaselineClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Imports (Adjust paths as needed) ---
sys.path.append('./hific')
sys.path.append('./hific/src')


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_baseline.png'):
    """
    Generates and saves a confusion matrix heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Baseline)')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")
    plt.close()


def synchronize_device(device):
    """
    Synchronizes the device for accurate timing.
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def main():
    # --- Config ---
    hific_checkpoint_path = './hific_low.pt'
    baseline_classifier_checkpoint_path = 'best_baseline_classifier.pth'
    dataset_path = './dataset'
    batch_size = 32
    num_classes = 6

    # Ensure these match your folder names exactly, alphabetically sorted usually
    class_names = ['B.subtilis', 'C.albicans',
                   'Contamination', 'E.coli', 'P.aeruginosa', 'S.aureus']

    # --- Hardware Optimization ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # --- Load HiFiC model ---
    class DummyLogger:
        def info(self, *args, **kwargs): pass

        def warning(self, *args, **kwargs): pass

    logger = DummyLogger()
    print("Loading HiFiC backbone...")
    hific_args.n_residual_blocks = 7
    hific_model = HiFiCModel(
        hific_args, logger, model_mode=ModelModes.EVALUATION)

    # Load Backbone Weights
    checkpoint = torch.load(hific_checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    try:
        hific_model.load_state_dict(new_state_dict, strict=.venv)
    except RuntimeError:
        hific_model.load_state_dict(new_state_dict, strict=False)

    # Freeze Backbone
    for param in hific_model.parameters():
        param.requires_grad = False
    hific_model.to(device)
    hific_model.eval()

    # --- Load Baseline Classifier ---
    print("Loading Classifier Head...")
    model = BaselineClassifier(hific_model, num_classes=num_classes)

    # Load Classifier Weights
    model.load_state_dict(torch.load(
        baseline_classifier_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Data ---
    # MUST MATCH TRAINING TRANSFORM EXACTLY
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = ImageFolder(os.path.join(
        dataset_path, 'test'), transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Evaluating on {len(test_dataset)} images...")

    # --- Evaluation Loop ---
    all_preds = []
    all_labels = []
    total_inference_time = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            total_inference_time += (end_time - start_time)


            _, predicted = torch.max(outputs, 1)

            # Store results on CPU for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics ---
    # 1. Accuracy
    acc = accuracy_score(all_labels, all_preds) * 100

    # 2. Timing
    num_images = len(test_dataset)

    avg_total_time_ms = (total_inference_time / num_images) * 1000
    fps_total = 1.0 / (total_inference_time / num_images)


    print("-" * 30)
    print(f"Global Accuracy: {acc:.2f}%")
    print("-" * 30)
    print("Inference Speed:")
    print(
        f"  - Total:          {avg_total_time_ms:.2f} ms/image ({fps_total:.1f} FPS)")
    print("-" * 30)

    # 3. Detailed Report (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 4. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)


if __name__ == '__main__':
    main()
