import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

import sys; from pathlib import Path; sys.path.append(str(Path(__file__).resolve().parents[2])); from helpers import create_tf_hub_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def apply_apa_with_upsized_delta(
    original_image: np.ndarray, 
    scale: float = 0.3, 
    target_label: int = 865,
    iters: int = 40
) -> np.ndarray:
    """
    Applies a Native TensorFlow Adversarial Patch to the image to force a target class.
    """
    logger.info(f"Applying Native Adversarial Patch Attack, scale {scale}, target {target_label}")

    orig_h, orig_w, _ = original_image.shape
    
    # Model expects 224x224
    small_img = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_AREA)
    x_small = small_img.astype(np.float32)[None] / 255.0

    classifier = create_tf_hub_classifier()
    model = getattr(classifier, "model", getattr(classifier, "_model", None))
    if model is None:
        logger.error("Could not extract native Keras model from classifier.")
        return original_image
    
    # Setup target
    num_classes = classifier.nb_classes
    y_target = np.zeros((1, num_classes), dtype=np.float32)
    y_target[0, target_label] = 1.0

    # Create spatial mask for the patch
    patch_size = max(16, int(224 * scale))
    start_y = np.random.randint(0, 224 - patch_size)
    start_x = np.random.randint(0, 224 - patch_size)
    
    mask = np.zeros((1, 224, 224, 3), dtype=np.float32)
    mask[:, start_y:start_y+patch_size, start_x:start_x+patch_size, :] = 1.0

    # Initialize patch with image pixels mixed with some random noise
    patch_noise = np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
    x_adv = x_small * (1 - mask) + patch_noise * mask
    
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    # Native PGD Loop specifically for the patch
    eps_step = 0.05
    for _ in range(iters):
        x_tensor = tf.convert_to_tensor(x_adv)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predictions = model(x_tensor)
            loss = loss_fn(y_target, predictions)
        
        grad = tape.gradient(loss, x_tensor)
        
        # We step in the direction of negative gradient to minimize target loss
        signed_grad = tf.sign(grad).numpy()
        x_adv = x_adv - eps_step * signed_grad * mask
        x_adv = np.clip(x_adv, 0.0, 1.0)
        
        # Ensure outside patch remains completely untouched
        x_adv = x_small * (1 - mask) + x_adv * mask

    # Extract the delta to scale it up
    delta_small = x_adv - x_small
    delta_small_0 = delta_small[0]
    
    # Precise high-resolution upscaling to prevent hard macro-blocking on the patch
    delta_small_contig = np.ascontiguousarray(delta_small_0, dtype=np.float32)
    delta_big = cv2.resize(delta_small_contig, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    # Slight blur over the upscaled patch to bleed it neatly into the shot
    delta_big = cv2.GaussianBlur(delta_big, (3, 3), 0)
    
    # Blend with native image 
    x_orig_float = original_image.astype(np.float32) / 255.0
    x_orig_adv = np.clip(x_orig_float + delta_big.astype(np.float32), 0.0, 1.0)
    
    return (x_orig_adv * 255.0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Apply Native Adversarial Patch Attack to an image.")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("output", type=str, help="Path to save output image")
    parser.add_argument("--scale", type=float, default=0.3, help="Patch scale (0.0 to 1.0)")
    parser.add_argument("--target", type=int, default=865, help="Target class ID")
    parser.add_argument("--iters", type=int, default=40, help="Iterations")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist.")
        return

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error(f"Failed to load image {input_path}")
        return

    adv_image = apply_apa_with_upsized_delta(image, scale=args.scale, target_label=args.target, iters=args.iters)
    
    cv2.imwrite(args.output, adv_image)
    logger.info(f"Native Adversarial patch image saved to {args.output}")

if __name__ == "__main__":
    main()
