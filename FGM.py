import cv2
import numpy as np
from art.attacks.evasion import FastGradientMethod
from helpers import create_tf_hub_classifier

def apply_fgm_with_upsized_delta(original_image, epsilon=0.06, eps_steps=0.02,  target_label=0):
    """
    Generates an adversarial delta at 224x224 using targeted FGM,
    upsizes that delta, and applies it to the original image to preserve detail.
    """
    print(f"FGM (Targeted) with upsampled delta to force class {target_label}.")

    # Convert original to float [0,1] for blending
    orig_h, orig_w, _ = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # Downsize image to 224×224 for the model
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    x_small = small_img.astype(np.float32)[None] / 255.0  # shape (1,224,224,3)

    #set up a target label (one-hot encoding)
    num_classes = 1001  # ImageNet has 1001 classes
    y_target = np.zeros((1, num_classes), dtype=np.float32)
    y_target[0, target_label] = 1.0  # Force classification as 'keeshond' (class 262)

    #run targeted FGM on the smaller image
    classifier = create_tf_hub_classifier()
    attack = FastGradientMethod(estimator=classifier, eps=epsilon, targeted=True, num_random_init=3, eps_step=eps_steps)
    x_adv_small = attack.generate(x=x_small, y=y_target)  # shape (1,224,224,3), in [0,1]

    # Compute small delta
    delta_small = x_adv_small - x_small  # shape (1,224,224,3)

    # Upsize delta to match original resolution
    delta_small_0 = delta_small[0]  # remove batch dimension
    delta_big = cv2.resize(delta_small_0, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Add the upscaled delta to the original float
    x_orig_adv = x_orig_float + delta_big
    x_orig_adv = np.clip(x_orig_adv, 0.0, 1.0) * 255.0
    x_orig_adv = x_orig_adv.astype(np.uint8)

    return x_orig_adv
