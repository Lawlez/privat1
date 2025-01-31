# Adversarial Attack Repository

This repository contains implementations of various adversarial attack methods on neural networks. It is designed to help researchers and practitioners explore adversarial robustness and test different attack strategies against deep learning models.

## Getting Started

The main entry point for this repository is **`privat1_tf.py`**. This script orchestrates the adversarial attacks, applies transformations to input images, and integrates various attack implementations.

### Files Overview

#### 1. `privat1_tf.py` (Main Script)
- Loads and preprocesses images.
- Applies multiple adversarial attack methods (FGM, PGD, Carlini-Wagner L2).
- Uses TensorFlow Hub models for classification.
- Includes additional image transformations such as pixel shifting, noise injection, and metadata manipulation.
- Outputs adversarially perturbed images with optional metadata stripping.

#### 2. `carlini_wagner.py`
- Implements the **Carlini & Wagner L2 (C&W) attack**.
- Generates adversarial perturbations using optimization-based techniques.
- The perturbations are computed on a downsized 224x224 image and upscaled to preserve detail before being applied to the original image.

#### 3. `FGM.py`
- Implements the **Fast Gradient Method (FGM)**.
- Perturbs input images along the gradient direction.
- Similar to the PGD method but with a single-step perturbation.
- Uses downscaled images for computation, then upsizes the perturbation to minimize resolution loss.

#### 4. `PGD.py`
- Implements the **Projected Gradient Descent (PGD) attack**.
- Iteratively updates adversarial perturbations with projected steps to remain within a constrained epsilon range.
- Preserves image detail by computing perturbations on a lower-resolution version and upscaling them before application.

#### 5. `helpers.py`
- Contains utility functions, including a TensorFlow Hub classifier loader.
- Loads a MobileNetV2-based ImageNet classifier.
- Wraps the classifier using the `art` (Adversarial Robustness Toolbox) framework for use in adversarial attacks.

## Running the Code

This repository is currently only compatible with macOS, as it relies on tensorflow-macos and tensorflow-metal for acceleration. Ensure that you have the required Python libraries installed before running the scripts:
```
pip install tensorflow-macos tensorflow-metal numpy torch art torchvision matplotlib tensorflow_hub
```
Then, to run an attack, execute:
```
python privat1_tf.py
```

## Contribution
Feel free to contribute by improving existing implementations, adding new adversarial attacks, or refining the evaluation metrics.

## License
This project is released under an open-source license. Use it for research and educational purposes only.

