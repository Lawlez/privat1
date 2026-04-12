import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from art.attacks.evasion import PixelAttack, ProjectedGradientDescent
import sys; from pathlib import Path; sys.path.append(str(Path(__file__).resolve().parents[2])); from helpers import create_tf_hub_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def apply_opa_with_upsized_delta(
    original_image: np.ndarray, 
    th: int = 5, 
    es: int = 50, 
    apply_pgd: bool = False
) -> np.ndarray:
    """
    Applies a One-Pixel Attack by modifying multiple pixels.
    """
    logger.info(f"Applying One-Pixel Attack, pixels: {th}, iterations: {es}")

    orig_h, orig_w, _ = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # Downsize for model
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_AREA)
    x_small = small_img.astype(np.float32)[None] / 255.0

    classifier = create_tf_hub_classifier()

    # Get original predictions to find a dynamic target class
    logits = classifier.predict(x_small)
    predicted_label = np.argmax(logits)
    top_5_classes = np.argsort(logits[0])[-5:]
    target_label = top_5_classes[-2] if top_5_classes[-2] != predicted_label else top_5_classes[-3]
    logger.info(f"Original prediction: {predicted_label}, Dynamically targeting: {target_label}")

    # One-Pixel Attack
    attack = PixelAttack(classifier, th=th, es=es)
    x_adv_small = attack.generate(x=x_small, y=np.array([target_label]))

    # Optional PGD step
    if apply_pgd:
        pgd_attack = ProjectedGradientDescent(classifier, eps=0.03, eps_step=0.01, max_iter=5, targeted=False)
        x_adv_small = pgd_attack.generate(x=x_adv_small)

    # Compute delta and upscale
    delta_small = x_adv_small - x_small
    delta_big = cv2.resize(delta_small[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    x_orig_adv = np.clip(x_orig_float + delta_big, 0, 1)
    return (x_orig_adv * 255.0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Apply One-Pixel Adversarial Attack to an image.")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("output", type=str, help="Path to save output image")
    parser.add_argument("--pixels", type=int, default=5, help="Number of pixels to modify")
    parser.add_argument("--iters", type=int, default=50, help="Search iterations")
    parser.add_argument("--pgd", action="store_true", help="Apply additional PGD noise")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist.")
        return

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error(f"Failed to load image {input_path}")
        return

    adv_image = apply_opa_with_upsized_delta(image, th=args.pixels, es=args.iters, apply_pgd=args.pgd)
    
    cv2.imwrite(args.output, adv_image)
    logger.info(f"One-pixel attack image saved to {args.output}")

if __name__ == "__main__":
    main()
