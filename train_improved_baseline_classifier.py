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

from improved_baseline_classifier import ImprovedBaselineClassifier


def main():
    # --- Config ---
    dataset_path = './dataset'
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10
    num_classes = 7
    quality = 3  # Minnen2018 quality level (1-8)
    dropout_rate = 0.5 # Dropout rate for the classifier head

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

    # --- Create Improved Baseline Classifier ---
    model = ImprovedBaselineClassifier(num_classes=num_classes, quality=quality, dropout_rate=dropout_rate)
    model.to(device)

    # --- Data & Augmentation ---
    # mbt2018_mean expects inputs in [0, 1]
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    # --- Training Setup ---
    # Only optimize parameters that require gradients (the classifier head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

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

        scheduler.step(avg_val_loss)

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save Best Model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_improved_baseline_classifier.pth')
            print(f"--> New best model saved with accuracy: {best_acc:.2f}%")

    print(f"Training finished. Best Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
