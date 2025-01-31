import cv2
import numpy as np
from art.attacks.evasion import FastGradientMethod
from helpers import create_tf_hub_classifier

def apply_fgm_with_upsized_delta(original_image, epsilon=0.04):
    """
    Generates an adversarial delta at 224x224 using FGM,
    upsizes that delta, and applies it to the original image
    to preserve detail.
    """
    print("FGM with upsampled delta to preserve detail.")

    # 1) Convert original to float [0,1] for blending
    orig_h, orig_w, _ = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # 2) Downsize image to 224×224 for the model
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    x_small = small_img.astype(np.float32)[None] / 255.0  # shape (1,224,224,3)

    # 3) Run FGM on the smaller image
    classifier = create_tf_hub_classifier()
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_adv_small = attack.generate(x_small)  # shape (1,224,224,3), in [0,1]

    # 4) Compute small delta
    delta_small = x_adv_small - x_small  # shape (1,224,224,3)

    # 5) Upsize delta to match original resolution
    delta_small_0 = delta_small[0]  # remove batch dimension
    delta_big = cv2.resize(delta_small_0, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 6) Add the upscaled delta to the original float
    x_orig_adv = x_orig_float + delta_big

    # 7) Clip to [0,1], convert back to uint8
    x_orig_adv = np.clip(x_orig_adv, 0.0, 1.0) * 255.0
    x_orig_adv = x_orig_adv.astype(np.uint8)

    return x_orig_adv
