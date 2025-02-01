import cv2
import numpy as np
from art.attacks.evasion import PixelAttack
from helpers import create_tf_hub_classifier

def apply_opa_with_upsized_delta(original_image, target_label=964, th=1, es=1):
    """
    Applies the One-Pixel Attack on an image.
    
    - Computes adversarial delta at 224x224.
    - Upscales the delta and applies it to the original.
    
    :return: Adversarial image in uint8 format (same shape as input).
    """
    print("Applying One-Pixel Attack.")

    # Convert original to float in [0,1] but keep a copy for final blending
    orig_h, orig_w, _ = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # Downsize to 224×224 for the model
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    x_small = small_img.astype(np.float32)[None] / 255.0  # shape (1,224,224,3)

    # Load classifier
    classifier = create_tf_hub_classifier()

    # Target preparation
    num_classes = classifier.nb_classes
    y_target = np.zeros((1, num_classes), dtype=np.float32)
    y_target[0, target_label] = 1.0

    # Apply One-Pixel Attack on the small version
    attack = PixelAttack(classifier, th=th, es=es)
    x_adv_small = attack.generate(x=x_small, y=y_target)  # shape (1,224,224,3)

    # Compute adversarial delta
    delta_small = x_adv_small - x_small  # shape (1,224,224,3)

    # Upscale delta to original size
    delta_big = cv2.resize(delta_small[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Apply adversarial perturbation
    x_orig_adv = np.clip(x_orig_float + delta_big, 0, 1)
    x_orig_adv = (x_orig_adv * 255.0).astype(np.uint8)

    return x_orig_adv
