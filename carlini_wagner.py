from art.attacks.evasion import CarliniL2Method
import cv2
import numpy as np
from helpers import create_tf_hub_classifier

def apply_cwl2_with_upsized_delta(original_image, confidence=1.0, max_iter=20):
    """
    Generates an adversarial delta at 224x224 using Carlini L2,
    upsizes that delta, applies to the original for minimal resolution loss.
    """
    print("Carlini L2 with upsampled delta to preserve detail.")

    # 1) Convert original to float [0,1]
    orig_h, orig_w, _ = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # 2) Downsize to 224×224
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    x_small = small_img.astype(np.float32)[None] / 255.0  # shape (1,224,224,3)

    # 3) Run Carlini L2 on the smaller version
    classifier = create_tf_hub_classifier()
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=confidence,
        max_iter=max_iter
    )
    x_adv_small = attack.generate(x_small)  # shape (1,224,224,3), in [0,1]

    # 4) delta
    delta_small = x_adv_small - x_small  # shape (1,224,224,3)

    # 5) Upsize the delta
    delta_small_0 = delta_small[0]
    delta_big = cv2.resize(delta_small_0, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 6) Apply the delta to the original float
    x_orig_adv = x_orig_float + delta_big

    # 7) Clip and convert
    x_orig_adv = np.clip(x_orig_adv, 0.0, 1.0) * 255.0
    x_orig_adv = x_orig_adv.astype(np.uint8)

    return x_orig_adv
