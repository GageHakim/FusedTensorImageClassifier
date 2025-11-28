# Fused and Baseline Classifiers for Latent Feature Spaces

This project provides an advanced image classification framework for identifying different types of bacteria. It features two primary models built in PyTorch:

1.  **`FusedClassifier`**: An advanced model that leverages multiple latent tensors from a pre-trained HiFiC (High-Fidelity Image Compression) model, using a gating mechanism to fuse them.
2.  **`BaselineClassifier`**: A simpler model that uses only the primary latent tensor from the HiFiC encoder, serving as a benchmark for the fusion approach.

The goal is to demonstrate that classification can be performed effectively—and perhaps more robustly—in the learned latent space of a powerful generative model rather than on raw pixels.

## Project Structure

```
.
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── hific/                  # HiFiC submodule
├── fused_classifier.py     # FusedClassifier model definition
├── baseline_classifier.py  # BaselineClassifier model definition
├── train_fused_classifier.py
├── train_baseline_classifier.py
├── evaluate_fused_classifier.py
├── evaluate_baseline_classifier.py
├── prepare_dataset.py      # Script to split data
├── requirements.txt
└── README.md
```

## Dataset

The dataset consists of images of the following six bacteria classes:
`B.subtilis`, `C.albicans`, `Contamination`, `E.coli`, `P.aeruginosa`, `S.aureus`.

The raw images are expected to be organized into subdirectories for each class. The `prepare_dataset.py` script handles splitting them into training, validation, and testing sets.

## Models

### 1. FusedClassifier

This model uses a pre-trained, frozen **HiFiC** model as a sophisticated feature extractor.

1.  **Feature Extraction**: The HiFiC encoder processes an input image and extracts two distinct latent tensors:
    *   `y`: A high-level representation of the image's content.
    *   `latent_scales`: A representation derived from the hyperprior, capturing information about textural complexity and local entropy.
2.  **Fusion**: A `GatedFusion` layer merges these two tensors with learnable weights, allowing the model to decide the optimal combination of content and texture features for the classification task.
3.  **Classification**: The resulting fused tensor is passed to a custom **`LatentResNet`** classification head (trained from scratch) to produce the final prediction.

### 2. BaselineClassifier

This model provides a standard for comparison to evaluate the effectiveness of the fusion strategy.

1.  **Feature Extraction**: It uses the same frozen HiFiC encoder to extract only the primary content tensor, `y`.
2.  **Classification**: The `y` tensor is fed directly into the same **`LatentResNet`** architecture used by the `FusedClassifier`. It does not use `latent_scales` or the `GatedFusion` layer.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch & TorchVision
- Dependencies from `requirements.txt`

### Installation

1.  Clone the repository and initialize the `hific` submodule:
    ```bash
    git clone --recurse-submodules https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to build the C++ extension for HiFiC. Navigate to the `hific` directory and run `python setup.py develop`.*

## Usage

### 1. Prepare the Dataset

If your dataset is not already split, run the preparation script. This will organize your raw images into `train`, `val`, and `test` directories.

```bash
python prepare_dataset.py
```

### 2. Train a Model

You can train either the fused or the baseline classifier. The scripts will load the pre-trained HiFiC model, train the designated classifier head, and save the best-performing weights.

**Train the Fused Classifier:**
```bash
python train_fused_classifier.py
```
*(Saves the best model to `best_fused_classifier.pth`)*

**Train the Baseline Classifier:**
```bash
python train_baseline_classifier.py
```
*(Saves the best model to `best_baseline_classifier.pth`)*

*Pre-trained HiFiC weights (`hific_low.pt`) can be downloaded from [this Google Drive folder](https://drive.google.com/drive/folders/1M9u6C53wyyjp519ZYfTBI3YEgUDyok1P?usp=sharing).*

### 3. Evaluate a Model

Evaluate your trained models on the test set to get accuracy, a detailed classification report, and a confusion matrix.

**Evaluate the Fused Classifier:**
```bash
python evaluate_fused_classifier.py
```
*(Loads `best_fused_classifier.pth` and saves `confusion_matrix.png`)*

**Evaluate the Baseline Classifier:**
```bash
python evaluate_baseline_classifier.py
```
*(Loads `best_baseline_classifier.pth` and saves `confusion_matrix_baseline.png`)*