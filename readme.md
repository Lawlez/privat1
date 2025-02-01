# Adversarial Attack Repository

This repository contains implementations of various adversarial attack methods on neural networks. It is designed to help researchers and practitioners explore adversarial robustness and test different attack strategies against deep learning models.


## Examples
Initial image:

![original](images/th3.jpeg)

Detection before:
```
🖼️ Image: privat1/images/th3.jpeg
  - suit (ID 835) -> Confidence: 6.3251
  - ski mask (ID 797) -> Confidence: 5.3537
  - abaya (ID 400) -> Confidence: 5.3193
  - trench coat (ID 870) -> Confidence: 5.1354
  - Windsor tie (ID 907) -> Confidence: 4.6823
❌ Forbidden object detected!
```

Obfuscated image:

![obfuscated](converted/examples/minimal_adv_all_cbbdddb1ae7a55a3.jpeg)

Detection After 
```
🖼️ Image: privat1/converted/examples/minimal_adv_all_cbbdddb1ae7a55a3.jpeg
  - jigsaw puzzle (ID 612) -> Confidence: 8.2029
  - fur coat (ID 569) -> Confidence: 6.1599
  - bearskin (ID 440) -> Confidence: 5.8124
  - chain mail (ID 491) -> Confidence: 5.3193
  - monastery (ID 664) -> Confidence: 5.2024
✅ Success: No forbidden objects detected.
```


## enhancing & analyisis
Examples and tools in /revert

Enhance1 (contrast enhancement):
![Enhance1](revert/enhance1_img.png)

Enhance2 (edge detection & thresholding):
![Enhance2](revert/enhance2_img.png)

Enhance3 (LSB analysis and frequency spectrum):
![Enhance3](revert/enhance3_img.png)

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
pip install -r requirements.txt
```
Then, to run an attack, execute:
```
python privat1_tf.py
```

## Contribution
Feel free to contribute by improving existing implementations, adding new adversarial attacks, or refining the evaluation metrics.

## License
This project is released under an open-source license. Use it for research and educational purposes only.

