
# VGG6 Image Classification on CIFAR-10


## 1. Project Overview

This project implements and evaluates a VGG6 Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset.

The main objective is to investigate the impact of different activation functions, optimizers, and batch sizes on classification performance.

Experiments are conducted using W&B sweeps to compare configurations, monitor training performance, and identify the best-performing model. The selected model is subsequently evaluated on the official CIFAR-10 test dataset.

**Final test accuracy: 88.01%.**

## 2. Dataset and Preprocessing

The CIFAR-10 dataset contains 60,000 colour images of size 32 × 32 pixels, distributed across 10 classes.

| Dataset split | Number of images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Testing | 10,000 |
| **Total** | **60,000** |

The original training dataset is divided into training and validation subsets using a reproducible random split.

### Preprocessing

- Conversion of images into PyTorch tensors.
- Normalization using CIFAR-10 channel statistics.
- Data augmentation applied to training images.
- Validation and test datasets use evaluation transformations without random augmentation.
- Fixed random seed for reproducibility.

## 3. VGG6 Model Architecture

The project implements a VGG6-style CNN consisting of six convolutional layers arranged into three convolutional blocks, followed by a fully connected classifier.

### 3.1 Layer-wise Architecture

| Layer | Operation | Output channels / neurons |
|---|---|---:|
| Input | RGB image (32 × 32) | 3 |
| Conv1 | 3 × 3 convolution, padding 1 | 64 |
| BatchNorm + Activation | Batch normalization + configurable activation | 64 |
| Conv2 | 3 × 3 convolution, padding 1 | 64 |
| BatchNorm + Activation | Batch normalization + configurable activation | 64 |
| Pool1 | 2 × 2 max pooling, stride 2 | 64 |
| Conv3 | 3 × 3 convolution, padding 1 | 128 |
| BatchNorm + Activation | Batch normalization + configurable activation | 128 |
| Conv4 | 3 × 3 convolution, padding 1 | 128 |
| BatchNorm + Activation | Batch normalization + configurable activation | 128 |
| Pool2 | 2 × 2 max pooling, stride 2 | 128 |
| Conv5 | 3 × 3 convolution, padding 1 | 256 |
| BatchNorm + Activation | Batch normalization + configurable activation | 256 |
| Conv6 | 3 × 3 convolution, padding 1 | 256 |
| BatchNorm + Activation | Batch normalization + configurable activation | 256 |
| Pool3 | 2 × 2 max pooling, stride 2 | 256 |
| Flatten | Flatten feature maps | 4096 |
| FC1 | Fully connected layer | 512 |
| Activation + Dropout | Configurable activation, dropout = 0.3 | 512 |
| FC2 | Classification output | 10 |

### 3.2 Architecture Flow

```text
Input: 3 × 32 × 32
        |
        v
Conv2D (3 → 64)
BatchNorm + Activation
Conv2D (64 → 64)
BatchNorm + Activation
MaxPool2D
        |
        v
Conv2D (64 → 128)
BatchNorm + Activation
Conv2D (128 → 128)
BatchNorm + Activation
MaxPool2D
        |
        v
Conv2D (128 → 256)
BatchNorm + Activation
Conv2D (256 → 256)
BatchNorm + Activation
MaxPool2D
        |
        v
Flatten (4096)
        |
Fully Connected (4096 → 512)
        |
Activation + Dropout (0.3)
        |
Fully Connected (512 → 10)
        |
        v
10 CIFAR-10 class predictions
```

### 3.3 Feature Map Dimensions

| Stage | Feature map dimensions |
|---|---|
| Input | 3 × 32 × 32 |
| Block 1 | 64 × 16 × 16 |
| Block 2 | 128 × 8 × 8 |
| Block 3 | 256 × 4 × 4 |
| Flatten | 4096 |
| Fully connected layer | 512 |
| Output | 10 |

### 3.4 Configurable Activation Functions

The implementation supports five activation functions:

- ReLU
- Sigmoid
- Tanh
- SiLU
- GELU

The selected activation function is applied after every convolutional layer and after the first fully connected layer.

The model contains approximately 3.25 million trainable parameters.

## 4. Experimental Setup and Results

Experiments were conducted to investigate the effect of activation functions and optimizers at two different batch sizes: 64 and 128.

The experiments used:
- Learning rate: 0.001
- Training epochs: 20
- Random seed: 42
- Dropout: 0.3
- Weight decay: 0.0001

### 4.1 Activation Function Comparison

In this experiment, Adam was kept fixed while the activation function was varied.

| Activation | Batch size 64 | Batch size 128 |
|---|---:|---:|
| ReLU | 87.34% | 87.94% |
| Sigmoid | 60.36% | 50.44% |
| Tanh | 80.14% | 80.24% |
| SiLU | 87.50% | **88.04%** |
| GELU | **88.58%** | 87.94% |

**Observations:**

- GELU achieved the highest validation accuracy of 88.58% with batch size 64.
- SiLU achieved the highest validation accuracy of 88.04% with batch size 128.
- ReLU and SiLU showed competitive performance across both batch sizes.
- Sigmoid performed considerably worse than the other activation functions.
- Batch size influenced the performance of different activation functions differently.

### 4.2 Optimizer Comparison

In this experiment, ReLU was kept fixed while the optimizer was varied.

| Optimizer | Batch size 64 | Batch size 128 |
|---|---:|---:|
| SGD | 85.18% | 84.44% |
| Nesterov | 85.72% | 84.34% |
| Adam | **87.72%** | 86.54% |
| Adagrad | 80.66% | 80.16% |
| RMSprop | 69.62% | 73.62% |
| Nadam | 87.70% | **86.66%** |

**Observations:**

- Adam achieved the highest validation accuracy of 87.72% with batch size 64.
- Nadam achieved the highest validation accuracy of 86.66% with batch size 128.
- Adam and Nadam performed very similarly at batch size 64.
- RMSprop showed the lowest validation accuracy in both experiments.
- Batch size 64 performed better for most optimizers, although the effect was not uniform.

### 4.3 Summary of Experimental Winners

| Experiment | Batch size | Best configuration | Validation accuracy |
|---|---:|---|---:|
| Activation sweep | 64 | GELU + Adam | **88.58%** |
| Activation sweep | 128 | SiLU + Adam | **88.04%** |
| Optimizer sweep | 64 | Adam + ReLU | **87.72%** |
| Optimizer sweep | 128 | Nadam + ReLU | **86.66%** |

The activation and optimizer sweeps were conducted independently, rather than as a complete factorial search across every possible combination.

The final selected configuration uses GELU activation and Adam optimizer with batch size 64. Adam was the fixed optimizer in the activation sweep that produced the best validation result.

## 5. Best Model Configuration

The best validation result was obtained using the following configuration:

| Hyperparameter | Value |
|---|---|
| Model architecture | VGG6 |
| Activation function | GELU |
| Optimizer | Adam |
| Batch size | 64 |
| Learning rate | 0.001 |
| Training epochs | 20 |
| Dropout | 0.3 |
| Weight decay | 0.0001 |
| Random seed | 42 |
| Best validation accuracy | **88.58%** |

The selected checkpoint corresponds to W&B run ID `epwq34m8`.

## 6. Weights & Biases Experiment Tracking

Weights & Biases was used to track model configurations, training metrics, validation metrics, and sweep results.

The experiments include:

- Activation function sweeps.
- Optimizer sweeps.
- Training loss and accuracy tracking.
- Validation loss and accuracy tracking.
- Hyperparameter comparison.
- Parallel-coordinate visualization.

## 7. Question 3: Experimental Visualizations

Six plots were generated using the W&B experiment data and training history.

| Figure | Visualization | File |
|---|---|---|
| Figure 1 | Parallel coordinates: configuration vs. accuracy | `01_parallel_coordinates.png` |
| Figure 2 | Validation accuracy vs. step (scatter plot) | `02_validation_accuracy_vs_step.png` |
| Figure 3 | Training loss | `03_training_loss.png` |
| Figure 4 | Training accuracy | `04_training_accuracy.png` |
| Figure 5 | Validation accuracy | `05_validation_accuracy.png` |
| Figure 6 | Validation loss | `06_validation_loss.png` |

All six figures are stored in `reports/plots/`.

The parallel-coordinate visualization compares experimental configurations and their validation accuracy. The remaining plots illustrate the model's training and validation behaviour over time.

## 8. Question 4: Final Model Performance

The selected model was evaluated on the official CIFAR-10 test dataset containing 10,000 images.

### 8.1 Test Results

| Evaluation metric | Result |
|---|---:|
| Total test samples | 10,000 |
| Correct predictions | 8,801 |
| Incorrect predictions | 1,199 |
| Test loss | 0.3561 |
| **Test top-1 accuracy** | **88.01%** |

### 8.2 Performance Analysis

The selected VGG6 model achieved 88.01% test accuracy with a test loss of 0.3561.

The best validation accuracy was 88.58%, resulting in a difference of 0.57 percentage points between validation and test accuracy.

This indicates reasonably consistent performance on unseen test samples.

The official test dataset was used for final evaluation rather than hyperparameter selection.

The evaluated checkpoint was:

`checkpoints/best_model_epwq34m8.pth`

## 9. Repository Structure

```text
CS6886W_VGG6_Assignment1/
│
├── configs/
│   ├── sweep_activation.yaml
│   ├── sweep_optimizer.yaml
│   └── sweep_batch_lr.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── engine.py
│   ├── train.py
│   └── evaluate.py
│
├── checkpoints/
│   └── best_model_epwq34m8.pth
│
├── reports/
│   ├── plots/
│   └── experiment_results.csv
│
├── main.py
└── README.md
```

This represents the intended project organization. Only files actually present in the final repository should be included.

## 10. Installation and Reproducibility

### Requirements

- Python 3.10 or later
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- PyYAML
- Weights & Biases

### Install dependencies

```bash
pip install torch torchvision numpy pandas matplotlib pyyaml wandb
```

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CS6886W_VGG6_Assignment1
```

### Configure W&B

```bash
wandb login
```

### Run experiments

The project provides a modular training pipeline and YAML configuration files for conducting experiments.

Check the available command-line arguments:

```bash
python main.py --help
```

Use the supported arguments and configuration files in `configs/` to reproduce the experiments.

### Evaluate the selected model

After downloading the checkpoint and ensuring the CIFAR-10 dataset is available, execute:

```bash
python -m src.evaluate
```

The evaluation script loads the selected GELU-based VGG6 checkpoint and reports the final test loss and top-1 accuracy.

## 11. Conclusion

This project presents a modular PyTorch implementation of VGG6 for CIFAR-10 image classification.

The experiments investigate the influence of activation functions, optimizers, and batch sizes on model performance.

GELU with Adam and batch size 64 achieved the best validation accuracy of 88.58%. The final selected model achieved **88.01% test accuracy** on the official CIFAR-10 test dataset.

The project demonstrates CNN implementation, reproducible dataset preparation, hyperparameter experimentation, W&B sweep tracking, model checkpointing, visualization, and independent test-set evaluation.
