import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import sys
import copy
from tqdm import tqdm

from scripts.fused_classifier import FusedClassifier
from hific.src.helpers import utils
from hific.default_config import ModelModes, hific_args
from hific.src.model import Model as HiFiCModel


def main():
    # --- Config ---
    hific_checkpoint_path = './hific_low.pt'
    dataset_path = './dataset'
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10
    num_classes = 6

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (Nvidia)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU (Warning: Slow)")
    pin_memory = True if device.type == 'cuda' else False
    print(f"Using device: {device}")

    # --- Load HiFiC model ---
    class DummyLogger:
        def info(self, *args, **kwargs): pass

        def warning(self, *args, **kwargs): pass

    logger = DummyLogger()

    print("Loading HiFiC model...")
    hific_args.n_residual_blocks = 7
    # Force evaluation mode for the backbone
    hific_model = HiFiCModel(hific_args, logger, model_mode=ModelModes.EVALUATION)

    checkpoint = torch.load(hific_checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Handle DataParallel prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Use strict=True first to ensure integrity, fall back only if necessary
    try:
        hific_model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed. Keys missing/unexpected: {e}")
        hific_model.load_state_dict(new_state_dict, strict=False)

    # FREEZE HiFiC Parameters (Critical for Transfer Learning)
    for param in hific_model.parameters():
        param.requires_grad = False

    hific_model.to(device)
    hific_model.eval()
    print("HiFiC model loaded and frozen.")

    # --- Create Fused Classifier ---
    # Ensure FusedClassifier properly registers hific_model
    model = FusedClassifier(hific_model, num_classes=num_classes)
    model.to(device)

    # --- Data & Augmentation ---
    # NOTE: HiFiC usually expects inputs in [-1, 1].
    # ToTensor gives [0, 1]. The Normalize((0.5,), (0.5,)) converts [0, 1] -> [-1, 1].
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Bacteria orientation is arbitrary
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, 'val'), transform=val_transform)

    # increased num_workers for speed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    # --- Training Setup ---
    # Only optimize parameters that require gradients (the classifier head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Scheduler to reduce LR if validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_acc = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for images, labels in pbar_train:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar_train.set_postfix({'Loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}")

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in pbar_val:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar_val.set_postfix({'Loss': loss.item()})


        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        # Step scheduler
        scheduler.step(avg_val_loss)

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save Best Model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_fused_classifier.pth')
            print(f"--> New best model saved with accuracy: {best_acc:.2f}%")

    print(f"Training finished. Best Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()