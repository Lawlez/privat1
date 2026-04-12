import argparse
import logging
from pathlib import Path
import random
import string
import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    return image

def apply_blur(image: np.ndarray, kernel_size: tuple = (3, 3)) -> np.ndarray:
    logger.info("Applying Gaussian Blur.")
    return cv2.GaussianBlur(image, kernel_size, 0)

def add_noise(image: np.ndarray, noise_level: int = 20) -> np.ndarray:
    logger.info(f"Adding Random Noise (level={noise_level}).")
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noise = np.random.randint(-noise_level, noise_level, (h, w, c), dtype='int16')
    return cv2.add(noisy_image, noise, dtype=cv2.CV_8U)

def pixelate(image: np.ndarray, pixel_size: int = 3) -> np.ndarray:
    logger.info(f"Applying Pixelation (size={pixel_size}).")
    h, w = image.shape[:2]
    temp = cv2.resize(image, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def generate_random_name(extension: str) -> str:
    random_name = ''.join(random.choices(string.hexdigits.lower(), k=16))
    return f"{random_name}.{extension}"

def remove_metadata(src_image: np.ndarray, output_path: Path):
    """Strip metadata by re-saving raw pixels via PIL."""
    image = Image.fromarray(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))
    data = list(image.getdata())
    stripped = Image.new(image.mode, image.size)
    stripped.putdata(data)
    stripped.save(output_path, "JPEG", quality=95)
    logger.info(f"Metadata-stripped image saved to {output_path}")

def protect_image(image_path: Path, output_dir: Path) -> Path:
    image = load_image(image_path)
    image = apply_blur(image)
    image = add_noise(image)
    image = pixelate(image)
    
    output_name = generate_random_name("jpg")
    output_path = output_dir / output_name
    remove_metadata(image, output_path)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="NBP - Noise, Blur, and Pixelation Protection")
    parser.add_argument("--input", type=str, default="images/", help="Input directory")
    parser.add_argument("--output", type=str, default="converted/NBP/", help="Output directory")
    
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
                protect_image(img_path, output_dir)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

if __name__ == "__main__":
    main()
