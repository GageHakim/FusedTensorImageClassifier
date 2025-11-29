import matplotlib.pyplot as plt
import seaborn as sns
import torch


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
