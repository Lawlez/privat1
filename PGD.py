import cv2
import numpy as np
from art.attacks.evasion import ProjectedGradientDescent
from helpers import create_tf_hub_classifier

def apply_pgd_with_upsized_delta(original_image, eps=0.06, eps_step=0.01, max_iter=10, target_label=964):
    """
    Computes adversarial delta at 224x224, then upsizes the delta and applies
    it to the original, preserving more detail than doing a full down->up.
    """
    print("PGD with upsampled delta to preserve detail.")
    
    #convert original to float in [0,1], but keep a copy for final blending
    orig_h, orig_w, c = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # downsize to 224×224 for model
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    x_small = small_img.astype(np.float32)[None] / 255.0  # shape (1,224,224,3)
    
    #target prep
    num_classes = 1001
    target_label = 964 #pizza
    y_target = np.zeros((1, num_classes), dtype=np.float32)
    y_target[0, target_label] = 1.0

    # run PGD on the small version
    classifier = create_tf_hub_classifier()
    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
        targeted=True,
        num_random_init=2 
    )
    x_adv_small = attack.generate(x=x_small, y=y_target)# shape (1,224,224,3) in [0,1]

    # upsize delta to original shape
    delta_small = x_adv_small - x_small  # shape (1,224,224,3)
    delta_big = cv2.resize(delta_small[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    x_orig_adv = x_orig_float + delta_big
    x_orig_adv = np.clip(x_orig_adv, 0, 1) * 255.0
    x_orig_adv = x_orig_adv.astype(np.uint8)

    return x_orig_adv