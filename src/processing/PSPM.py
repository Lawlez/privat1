import argparse
import logging
from pathlib import Path
import random
import string
import cv2
import numpy as np
from PIL import Image
from art.attacks.evasion import FastGradientMethod
import sys; from pathlib import Path; sys.path.append(str(Path(__file__).resolve().parents[2])); from helpers import create_tf_hub_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    return image

def apply_adversarial_noise(image: np.ndarray, epsilon: float = 0.04) -> np.ndarray:
    """Applies adversarial noise using a real classifier and upscale-delta technique."""
    logger.info(f"Applying FGM Adversarial Noise (eps={epsilon})")
    orig_h, orig_w, _ = image.shape
    x_orig_float = image.astype(np.float32) / 255.0

    # Model resolution
    small_img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    x_small = small_img.astype(np.float32)[None] / 255.0

    classifier = create_tf_hub_classifier()
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_adv_small = attack.generate(x=x_small)

    # Upscale delta
    delta_small = x_adv_small - x_small
    delta_big = cv2.resize(delta_small[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    x_orig_adv = np.clip(x_orig_float + delta_big, 0.0, 1.0)
    adv_image = (x_orig_adv * 255.0).astype(np.uint8)
    
    # Slight blur to reduce visible artifacts
    return cv2.GaussianBlur(adv_image, (3, 3), 0)

def apply_pixel_shift(image: np.ndarray, shift_amount: int = 4) -> np.ndarray:
    logger.info("Applying Pixel Shift.")
    shifted_image = image.copy()
    h, w, _ = shifted_image.shape
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if x + shift_amount < w and y + shift_amount < h:
                shifted_image[y, x] = image[(y + shift_amount) % h, (x + shift_amount) % w]
    return shifted_image

def apply_pixel_pattern_mask(image: np.ndarray, pattern_size: int = 4, opacity: float = 0.2) -> np.ndarray:
    logger.info("Applying Pixel Pattern Mask.")
    h, w, c = image.shape
    masked_image = image.copy().astype(np.float32)
    pattern = (np.random.randint(0, 2, (pattern_size, pattern_size, c), dtype='uint8') * 50).astype(np.float32)
    
    for y in range(0, h, pattern_size):
        for x in range(0, w, pattern_size):
            h_end, w_end = min(y + pattern_size, h), min(x + pattern_size, w)
            h_len, w_len = h_end - y, w_end - x
            region = masked_image[y:h_end, x:w_end]
            blend = cv2.addWeighted(region, 1 - opacity, pattern[:h_len, :w_len], opacity, 0)
            masked_image[y:h_end, x:w_end] = blend
            
    return masked_image.astype(np.uint8)

def apply_random_perspective_transform(image: np.ndarray, margin: float = 0.02) -> np.ndarray:
    """Random perspective warp to shift corners and break bounding-box detection."""
    logger.info("Applying Random Perspective Transform.")
    h, w = image.shape[:2]
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    def rand_pt(x, y):
        mx, my = int(w * margin), int(h * margin)
        return [x + random.randint(-mx, mx), y + random.randint(-my, my)]

    pts2 = np.float32([rand_pt(0,0), rand_pt(w,0), rand_pt(0,h), rand_pt(w,h)])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def embed_distractor_assets(image: np.ndarray, assets_path: Path) -> np.ndarray:
    """Blends tiny distractor images into random spots."""
    if not assets_path.exists():
        logger.debug("Assets path not found, skipping distractor embedding.")
        return image

    logger.info("Embedding distractor image assets.")
    h, w, _ = image.shape
    assets = [f for f in assets_path.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    
    if not assets: return image

    for _ in range(15): # Blend 15 random assets
        asset_file = random.choice(assets)
        asset_img = cv2.imread(str(asset_file))
        if asset_img is None: continue
        
        # Resize to tiny distractor patch
        patch_size = random.randint(16, 32)
        asset_patch = cv2.resize(asset_img, (patch_size, patch_size))
        
        y_off = random.randint(0, h - patch_size)
        x_off = random.randint(0, w - patch_size)
        
        roi = image[y_off:y_off+patch_size, x_off:x_off+patch_size].astype(np.float32) / 255.0
        patch = asset_patch.astype(np.float32) / 255.0
        
        # Soft blend (Screen mode)
        blended = 1 - (1 - roi) * (1 - patch)
        blended = (blended * 0.4 + roi * 0.6) # 40% distractor, 60% original
        image[y_off:y_off+patch_size, x_off:x_off+patch_size] = (blended * 255.0).astype(np.uint8)

    return image

def generate_random_name(extension: str) -> str:
    random_name = ''.join(random.choices(string.hexdigits.lower(), k=16))
    return f"{random_name}.{extension}"

def remove_metadata(image_mat: np.ndarray, output_path: Path):
    """Stripps metadata by re-saving pixel data via PIL."""
    image = Image.fromarray(cv2.cvtColor(image_mat, cv2.COLOR_BGR2RGB))
    # data reconstruction to strip headers
    data = list(image.getdata())
    stripped = Image.new(image.mode, image.size)
    stripped.putdata(data)
    stripped.save(output_path, "JPEG", quality=95)
    logger.info(f"Metadata-stripped image saved to {output_path}")

def protect_image(image_path: Path, output_dir: Path, eps: float) -> Path:
    image = load_image(image_path)
    
    image = apply_adversarial_noise(image, epsilon=eps)
    image = apply_pixel_shift(image)
    image = apply_pixel_pattern_mask(image)
    
    output_name = generate_random_name("jpg")
    output_path = output_dir / output_name
    remove_metadata(image, output_path)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="PSPM - Pixel Shift Pattern Masking Protection")
    parser.add_argument("--input", type=str, default="images/", help="Input directory")
    parser.add_argument("--output", type=str, default="converted/PSPM/", help="Output directory")
    parser.add_argument("--eps", type=float, default=0.04, help="Adversarial epsilon")
    
    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input path {input_dir} not found.")
        return

    for img_path in input_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            try:
                protect_image(img_path, output_dir, args.eps)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

if __name__ == "__main__":
    main()
