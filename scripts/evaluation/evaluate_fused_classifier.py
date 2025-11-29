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

from scripts.fused_classifier import FusedClassifier
from scripts.evaluation_utils import plot_confusion_matrix, synchronize_device
from hific.src.helpers import utils
from hific.default_config import ModelModes, hific_args
from hific.src.model import Model as HiFiCModel


def evaluate_fused():
    # --- Config ---
    hific_checkpoint_path = './hific_low.pt'
    fused_classifier_checkpoint_path = 'best_fused_classifier.pth'
    dataset_path = './dataset'
    batch_size = 32
    num_classes = 6

    # Ensure these match your folder names exactly, alphabetically sorted usually
    class_names = ['B.subtilis', 'C.albicans', 'Contamination', 'E.coli', 'P.aeruginosa', 'S.aureus']

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
    hific_model = HiFiCModel(hific_args, logger, model_mode=ModelModes.EVALUATION)

    # Load Backbone Weights
    checkpoint = torch.load(hific_checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    try:
        hific_model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError:
        hific_model.load_state_dict(new_state_dict, strict=False)

    # Freeze Backbone
    for param in hific_model.parameters():
        param.requires_grad = False
    hific_model.to(device)
    hific_model.eval()

    # --- Load Fused Classifier ---
    print("Loading Classifier Head...")
    # Make sure this matches the training script (LatentResNet vs EfficientNet)
    model = FusedClassifier(hific_model, num_classes=num_classes)

    # Load Classifier Weights
    # We use strict=False here just in case of minor version diffs,
    # but strictly speaking, it should match perfectly.
    model.load_state_dict(torch.load(fused_classifier_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Data ---
    # MUST MATCH TRAINING TRANSFORM EXACTLY
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Evaluating on {len(test_dataset)} images...")

    # --- Evaluation Loop ---
    all_preds = []
    all_labels = []
    total_extraction_time = 0.0
    total_classification_time = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 1. Time Feature Extraction
            synchronize_device(device)
            start_extract = time.time()

            features = model.extract_features(images)

            synchronize_device(device)
            end_extract = time.time()
            total_extraction_time += (end_extract - start_extract)

            # 2. Time Classification
            synchronize_device(device)
            start_classify = time.time()

            outputs = model.classify_features(features)

            synchronize_device(device)
            end_classify = time.time()
            total_classification_time += (end_classify - start_classify)

            _, predicted = torch.max(outputs, 1)

            # Store results on CPU for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Metrics ---
    # 1. Accuracy
    acc = accuracy_score(all_labels, all_preds) * 100

    # 2. Timing
    total_time = total_extraction_time + total_classification_time
    num_images = len(test_dataset)

    avg_total_time_ms = (total_time / num_images) * 1000
    avg_extraction_time_ms = (total_extraction_time / num_images) * 1000
    avg_classification_time_ms = (total_classification_time / num_images) * 1000

    fps_total = 1.0 / (total_time / num_images) if total_time > 0 else 0
    fps_extraction = 1.0 / (total_extraction_time / num_images) if total_extraction_time > 0 else 0
    fps_classification = 1.0 / (total_classification_time / num_images) if total_classification_time > 0 else 0

    print("-" * 30)
    print(f"Global Accuracy: {acc:.2f}%")
    print("-" * 30)
    print("Inference Speed Breakdown:")
    print(f"  - Feature Extraction:    {avg_extraction_time_ms:.2f} ms/image ({fps_extraction:.1f} FPS)")
    print(f"  - Classification: {avg_classification_time_ms:.2f} ms/image ({fps_classification:.1f} FPS)")
    print(f"  - Total:          {avg_total_time_ms:.2f} ms/image ({fps_total:.1f} FPS)")
    print("-" * 30)

    # 3. Detailed Report (Precision, Recall, F1)
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))


    # 4. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_path='fused_confusion_matrix.png')

    return {
        'accuracy': acc,
        'avg_total_time_ms': avg_total_time_ms,
        'avg_extraction_time_ms': avg_extraction_time_ms,
        'avg_classification_time_ms': avg_classification_time_ms,
        'fps_total': fps_total,
        'fps_extraction': fps_extraction,
        'fps_classification': fps_classification,
        'report': report,
    }


if __name__ == '__main__':
    evaluate_fused()