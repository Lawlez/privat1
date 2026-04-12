import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from art.attacks.evasion import SpatialTransformation
import sys; from pathlib import Path; sys.path.append(str(Path(__file__).resolve().parents[2])); from helpers import create_tf_hub_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def apply_staa_with_upsized_delta(
    original_image: np.ndarray, 
    max_translation: float = 3.0, 
    max_rotation: float = 2.0, 
    target_label: int = 262
) -> np.ndarray:
    """
    Applies Spatial Transformation Adversarial Attack.
    """
    logger.info(f"Applying STAA, trans: {max_translation}, rot: {max_rotation}, target: {target_label}")

    orig_h, orig_w, _ = original_image.shape
    x_orig_float = original_image.astype(np.float32) / 255.0

    # Downsize for model
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_AREA)
    x_small = small_img.astype(np.float32)[None] / 255.0

    classifier = create_tf_hub_classifier()
    num_classes = classifier.nb_classes
    y_target = np.zeros((1, num_classes), dtype=np.float32)
    y_target[0, target_label] = 1.0

    # STAA Attack
    attack = SpatialTransformation(
        classifier, 
        max_translation=max_translation, 
        max_rotation=max_rotation, 
        num_translations=5, 
        num_rotations=5
    )
    x_adv_small = attack.generate(x=x_small, y=y_target)

    # Compute delta and upscale
    delta_small = x_adv_small - x_small
    delta_big = cv2.resize(delta_small[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    x_orig_adv = np.clip(x_orig_float + delta_big, 0, 1)
    return (x_orig_adv * 255.0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Apply Spatial Transformation Adversarial Attack to an image.")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("output", type=str, help="Path to save output image")
    parser.add_argument("--trans", type=float, default=10.0, help="Max translation")
    parser.add_argument("--rot", type=float, default=10.0, help="Max rotation")
    parser.add_argument("--target", type=int, default=262, help="Target class ID")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist.")
        return

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error(f"Failed to load image {input_path}")
        return

    adv_image = apply_staa_with_upsized_delta(
        image, 
        max_translation=args.trans, 
        max_rotation=args.rot, 
        target_label=args.target
    )
    
    cv2.imwrite(args.output, adv_image)
    logger.info(f"Spatial transformation attack image saved to {args.output}")

if __name__ == "__main__":
    main()
