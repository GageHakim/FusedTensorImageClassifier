# Bacteria Image Classifier with Fused Latent Features

This project is an advanced image classifier for different types of bacteria, built using a custom `FusedClassifier` model in PyTorch.

## Project Structure
```
.
├── dataset
│   ├── bacteria_resized_224  (raw data)
│   ├── train
│   ├── val
│   └── test
├── hific                     (HiFiC submodule)
├── fused_classifier.py       (Model definition)
├── train_fused_classifier.py (Training script)
├── evaluate_fused_classifier.py (Evaluation script)
├── prepare_dataset.py
├── requirements.txt
└── README.md
```

## Dataset

The dataset consists of images of the following bacteria:
*   `B.subtilis`
*   `C.albicans`
*   `Contamination`
*   `E.coli`
*   `P.aeruginosa`
*   `S.aureus`

The raw images are expected to be in the `dataset/bacteria_resized_224` directory, organized into subdirectories for each class.

## Model: FusedClassifier

This project uses a `FusedClassifier` that leverages a pre-trained, frozen **HiFiC** (High-Fidelity Image Compression) model as a feature extractor.

1.  **Feature Extraction**: The HiFiC model processes the input images and extracts two latent tensors:
    *   `y`: A representation of the image's content.
    *   `latent_scales`: A representation of the image's entropy and textural complexity.
2.  **Fusion**: A `GatedFusion` layer merges these two tensors, allowing the model to learn the optimal combination of content and texture features.
3.  **Classification**: The fused tensor is passed to an **EfficientNet** classification head (initialized from scratch) to produce the final prediction.

This approach allows the model to learn from the rich, compressed feature space of the HiFiC model rather than from raw pixels.

## Getting Started

### Prerequisites

*   Python 3.x
*   PyTorch
*   TorchVision
*   and other dependencies listed in `requirements.txt`

### Installation

1.  Clone the repository and initialize the `hific` submodule:
    ```bash
    git clone --recurse-submodules https://github.com/GageHakim/ImageClassifierEfficientNet.git
    cd ImageClassifierEfficientNet
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    You may also need to build the C++ extension for HiFiC, which can be done by running `python setup.py develop` inside the `hific` directory.

## Usage

### 1. Prepare the Dataset

If the dataset has not been split, run the `prepare_dataset.py` script from the root directory. This will split the raw images from `dataset/bacteria_resized_224` into `dataset/train`, `dataset/val`, and `dataset/test` directories.

```bash
python prepare_dataset.py
```

### 2. Train the Model

To train the `FusedClassifier`, run the `train_fused_classifier.py` script. The script will:
*   Load the pre-trained HiFiC model (`hific_low.pt`).
*   Train the `FusedClassifier`, updating only the fusion layer and the EfficientNet head.
*   Display progress bars for training and validation.
*   Save the best-performing model to `best_fused_classifier.pth`.

```bash
python train_fused_classifier.py
```

### 3. Evaluate the Model

To evaluate the best trained model on the test set, run the `evaluate_fused_classifier.py` script.

```bash
python evaluate_fused_classifier.py
```

This will load `best_fused_classifier.pth` and print the final accuracy and average inference time.