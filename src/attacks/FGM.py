import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from art.attacks.evasion import FastGradientMethod
import sys; from pathlib import Path; sys.path.append(str(Path(__file__).resolve().parents[2])); from helpers import create_tf_hub_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def apply_fgm_with_upsized_delta(
    original_image: np.ndarray, 
    epsilon: float = 0.06, 
    target_label: int = 0
) -> np.ndarray:
    """
    Generates an adversarial delta at 224x224 using targeted FGM,
    upsizes that delta, and applies it to the original image to preserve detail.
    """
    logger.info(f"Applying FGM (Targeted) to force class {target_label} with epsilon {epsilon}")

    # Convert original to float [0,1] for blending
    orig_h, orig_w, _ = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # Downsize image to 224×224 for the model
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_AREA)
    x_small = small_img.astype(np.float32)[None] / 255.0  # shape (1,224,224,3)

    # Set up a target label (one-hot encoding)
    classifier = create_tf_hub_classifier()
    num_classes = classifier.nb_classes
    y_target = np.zeros((1, num_classes), dtype=np.float32)
    y_target[0, target_label] = 1.0

    # Run targeted FGM on the smaller image
    attack = FastGradientMethod(estimator=classifier, eps=epsilon, targeted=True)
    x_adv_small = attack.generate(x=x_small, y=y_target)

    # Compute small delta
    delta_small = x_adv_small - x_small
    delta_small_0 = delta_small[0]

    # Upsize delta to match original resolution
    delta_big = cv2.resize(delta_small_0, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Add the upscaled delta to the original float
    x_orig_adv = x_orig_float + delta_big
    x_orig_adv = np.clip(x_orig_adv, 0.0, 1.0) * 255.0
    
    return x_orig_adv.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Apply FGM Adversarial Attack to an image.")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("output", type=str, help="Path to save output image")
    parser.add_argument("--eps", type=float, default=0.06, help="Epsilon (attack strength)")
    parser.add_argument("--target", type=int, default=262, help="Target class ID (default 262: keeshond)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist.")
        return

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error(f"Failed to load image {input_path}")
        return

    adv_image = apply_fgm_with_upsized_delta(image, epsilon=args.eps, target_label=args.target)
    
    cv2.imwrite(args.output, adv_image)
    logger.info(f"Adversarial image saved to {args.output}")

if __name__ == "__main__":
    main()
