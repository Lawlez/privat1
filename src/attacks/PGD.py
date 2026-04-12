import argparse
import logging
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
from art.attacks.evasion import ProjectedGradientDescent
import sys; from pathlib import Path; sys.path.append(str(Path(__file__).resolve().parents[2])); from helpers import create_tf_hub_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def apply_pgd_with_upsized_delta(
    original_image: np.ndarray, 
    epsilon: float = 0.06, 
    eps_step: float = None, 
    max_iter: int = 40, 
    target_label: int = 865,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Generates an adversarial delta at 224x224 using targeted PGD,
    upsizes that delta, and applies it to the original image.
    Uses INTER_CUBIC for high-resolution stability.
    """
    logger.info(f"Applying PGD (Targeted) to force class {target_label} with eps {epsilon}")

    orig_h, orig_w, _ = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # Model resolution
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_AREA)
    x_small = small_img.astype(np.float32)[None] / 255.0

    # Classifier
    classifier = create_tf_hub_classifier()
    num_classes = classifier.nb_classes
    y_target = np.zeros((1, num_classes), dtype=np.float32)
    y_target[0, target_label] = 1.0

    # Dynamic step sizing if none provided
    if eps_step is None:
        eps_step = epsilon / 5.0

    # PGD Attack
    attack = ProjectedGradientDescent(
        estimator=classifier, 
        eps=epsilon, 
        eps_step=eps_step, 
        max_iter=max_iter, 
        targeted=True
    )
    x_adv_small = attack.generate(x=x_small, y=y_target)

    # Clean any NaN/Inf failures caused by steep numerical gradients
    delta_small = x_adv_small - x_small
    delta_small = np.nan_to_num(delta_small, nan=0.0, posinf=0.0, neginf=0.0)
    delta_small = np.clip(delta_small, -epsilon, epsilon).astype(np.float32)
    
    # Delta scaling with cubic interpolation to softly blend adversarial patterns
    delta_small_contig = np.ascontiguousarray(delta_small[0], dtype=np.float32)
    delta_big = cv2.resize(delta_small_contig, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    # Slight blur to fully blend sharp boundaries
    delta_big = cv2.GaussianBlur(delta_big, (3, 3), 0)
    
    # Blend with original image using alpha for subtlety
    x_orig_adv = np.clip(x_orig_float + (delta_big.astype(np.float32) * alpha), 0.0, 1.0)
    return (x_orig_adv * 255.0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Apply PGD Adversarial Attack to an image.")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("output", type=str, help="Path to save output image")
    parser.add_argument("--eps", type=float, default=0.06, help="Epsilon (attack strength)")
    parser.add_argument("--step", type=float, default=0.01, help="Epsilon step")
    parser.add_argument("--iters", type=int, default=15, help="Max iterations")
    parser.add_argument("--target", type=int, default=865, help="Target class ID")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist.")
        return

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error(f"Failed to load image {input_path}")
        return

    adv_image = apply_pgd_with_upsized_delta(
        image, 
        epsilon=args.eps, 
        eps_step=args.step, 
        max_iter=args.iters, 
        target_label=args.target
    )
    
    cv2.imwrite(args.output, adv_image)
    logger.info(f"Adversarial image saved to {args.output}")

if __name__ == "__main__":
    main()