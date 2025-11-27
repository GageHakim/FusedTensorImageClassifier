# Bacteria Image Classifier

This project is an image classifier for different types of bacteria, built using EfficientNet and PyTorch.

## Dataset

The dataset consists of images of the following bacteria:

*   B.subtilis
*   C.albicans
*   Contamination
*   E.coli
*   P.aeruginosa
*   S.aureus

The dataset is organized into `train`, `val`, and `test` directories. The `prepare_dataset.py` script can be used to split the data into these directories.

## Model

The classifier uses the EfficientNet architecture, implemented in PyTorch using the `EfficientNet-PyTorch` library.

## Getting Started

### Prerequisites

*   Python 3.x
*   PyTorch
*   TorchVision
*   NumPy
*   scikit-learn

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/GageHakim/ImageClassifierEfficientNet.git
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Prepare the dataset:**
    Run the `prepare_dataset.py` script to split the dataset into training, validation, and testing sets.

2.  **Train the model:**
    Run the `train.py` script in the `EfficientNet-PyTorch` directory.
    ```bash
    python EfficientNet-PyTorch/train.py
    ```
