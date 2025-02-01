import cv2
import numpy as np
import random
from art.attacks.evasion import PixelAttack, ProjectedGradientDescent
from helpers import create_tf_hub_classifier

def apply_opa_with_upsized_delta(original_image, th=5, es=50, apply_pgd=False):
    """
    Applies a stronger One-Pixel Attack by:
    - Targeting a misclassification dynamically.
    - Modifying multiple pixels (th > 1).
    - Running evolutionary search with more iterations (es > 30).
    - Optionally adding a PGD step for extra perturbation.
    
    :return: Adversarial image (H, W, 3) in uint8 format.
    """
    print("Applying One-Pixel Attack.")

    orig_h, orig_w, _ = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # Downsize for model
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    x_small = small_img.astype(np.float32)[None] / 255.0

    classifier = create_tf_hub_classifier()

    # Get original predictions
    logits = classifier._model(x_small).numpy()
    predicted_label = np.argmax(logits)
    top_5_classes = np.argsort(logits[0])[-5:]
    target_label = top_5_classes[-2] if top_5_classes[-2] != predicted_label else top_5_classes[-3]
    print(f"Original prediction: {predicted_label}, Targeting: {target_label}")

    # Target preparation
    num_classes = classifier.nb_classes
    y_target = np.zeros((1, num_classes), dtype=np.float32)
    y_target[0, target_label] = 1.0

    # One-Pixel Attack
    attack = PixelAttack(classifier, th=th, es=es)
    x_adv_small = attack.generate(x=x_small, y=y_target)

    # Apply PGD after One-Pixel Attack (optional)
    if apply_pgd:
        pgd_attack = ProjectedGradientDescent(
            classifier,
            eps=0.03,  # Small noise
            eps_step=0.01,
            max_iter=5,
            targeted=False
        )
        x_adv_small = pgd_attack.generate(x=x_adv_small)

    # Compute delta and upscale
    delta_small = x_adv_small - x_small
    delta_big = cv2.resize(delta_small[0], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Apply adversarial perturbation
    x_orig_adv = np.clip(x_orig_float + delta_big, 0, 1)
    x_orig_adv = (x_orig_adv * 255.0).astype(np.uint8)

    return x_orig_adv
