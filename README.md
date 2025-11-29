# Fused Tensor Image Classifier

This project provides a framework for training and evaluating various image classification models, with a focus on comparing standard classifiers to innovative models that leverage "fused" tensors from a compression model's latent space. It is built to be easily extensible for comparing different model architectures.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
- [Models in Detail](#models-in-detail)
- [Scripts Overview](#scripts-overview)

## Project Structure

```
.
├── efficientnet_pytorch/  # EfficientNet model implementation
├── hific/                 # HiFiC compression model (submodule)
├── scripts/
│   ├── training/          # Scripts to train each classifier
│   ├── evaluation/        # Scripts to evaluate each classifier
│   ├── prepare_dataset.py # Script to process and split the dataset
│   └── predict.py         # Script for making predictions with a trained model
├── main.py                # Main script to run all evaluations and compare models
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

1. **Clone the repository with submodules:**

    ```bash
    git clone --recurse-submodules https://github.com/your-username/FusedTensorImageClassifier.git
    cd FusedTensorImageClassifier
    ```

2. **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Build the C++ extension for HiFiC:**

    The `hific` submodule requires a C++ extension to be built.

    ```bash
    cd hific
    python setup.py install
    cd ..
    ```

    > **Note:** If you encounter build issues, you may need to install C++ build tools. On macOS, this can be done by installing Xcode Command Line Tools. On Linux, you may need to install `build-essential`.

## Dataset Preparation

The project expects a dataset organized in the following structure:

```
dataset/
├── train/
│   ├── class1/
│   │   ├── img1.png
│   │   └── ...
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

You can use the `scripts/prepare_dataset.py` script to automate the process of splitting your data into training and validation sets. See the [Scripts Overview](#scripts-overview) section for more details.

## Usage

### Training Models

To train a model, you need to configure and run the corresponding script in `scripts/training/`. For example, to train the Fused Classifier:

1. **Edit the training script:** Open `scripts/training/train_fused_classifier.py` and modify the configuration variables at the top of the `main()` function, such as `hific_checkpoint_path`, `dataset_path`, `batch_size`, `learning_rate`, etc.

2. **Run the script:**

    ```bash
    python scripts/training/train_fused_classifier.py
    ```

The other training scripts follow a similar pattern. The trained model will be saved as a `.pth` file in the root directory.

### Evaluating Models

The `main.py` script is the main entry point for evaluating and comparing all the implemented models. It will run the evaluation for each classifier and print a summary table with performance metrics.

```bash
python main.py
```

The script will output:
- A description of each model.
- Tables comparing Accuracy, Inference Speed (ms/image), and Throughput (FPS).
- Confusion matrices for each model, saved as PNG files in the root directory.

You can also run individual evaluation scripts from `scripts/evaluation/`.

## Models in Detail

This project compares the following models:

- **Baseline Classifier:** A simple CNN trained from scratch on the target dataset. It serves as a performance baseline.

- **Improved Baseline:** This model uses a pre-trained `Minnen2018-Mean` image compression model as a frozen feature extractor. The latent tensor `y` from the compression model is fed into a custom lightweight ResNet for classification. This tests the raw feature quality of the pre-trained compression model.

- **Fused Classifier:** This model enhances the feature representation by fusing the main latent tensor `y` from the HiFiC compression model with a signal from its hyperprior. The hyperprior's role is to estimate the spatial distribution of bits, effectively acting as an **attention mechanism**. The `latent_scales` tensor from the hyperprior, which represents the uncertainty or importance of different parts of the latent space, is fused with `y` using a `GatedFusion` module. This allows the model to focus on more informative features before classification by the custom ResNet.

- **Mean Fused Classifier:** Similar to the Fused Classifier, this model uses a fusion approach. However, it employs the `mbt2018_mean` compression model as its backbone. It fuses the main latent tensor `y` with the `scales_hat` from the hyperprior, which also represents the bit distribution. The fused tensor is then classified by the same lightweight ResNet. This model explores a different pre-trained compression model's feature and hyperprior signal quality.

- **EfficientNet:** A standard, off-the-shelf image classification model (EfficientNet-B0) known for its high accuracy and efficiency. It is fine-tuned on the target dataset and serves as a strong, conventional benchmark.

## Scripts Overview

### `scripts/training/`

This directory contains scripts to train each of the classifiers.

- `train_baseline_classifier.py`: Trains the simple CNN.
- `train_improved_baseline_classifier.py`: Trains the Improved Baseline model.
- `train_fused_classifier.py`: Trains the Fused Classifier. Requires a pre-trained HiFiC model checkpoint.
- `train_mean_fused_classifier.py`: Trains the Mean Fused Classifier.
- `train.py`: A generic training script, likely for EfficientNet.

To run any of these scripts, you first need to **edit the script** to set the correct paths for the dataset and any required checkpoints, as well as to configure hyperparameters like learning rate, batch size, and number of epochs.

### `scripts/evaluation/`

This directory contains scripts to evaluate the performance of each trained classifier. These are called by `main.py`.

- `evaluate_baseline_classifier.py`
- `evaluate_improved_baseline_classifier.py`
- `evaluate_fused_classifier.py`
- `evaluate_mean_fused_classifier.py`
- `evaluate_efficientnet.py`

### Other Scripts

- **`scripts/prepare_dataset.py`**: This script is used to process a raw dataset into the format required for training. It reads a source directory containing images and `.json` files with bounding box annotations. It crops the images, resizes them with padding, and splits them into `train`, `val`, and `test` sets. You will need to edit the paths in this script to match your dataset location.

    ```bash
    python scripts/predict.py <path_to_your_image.jpg>
    ```
