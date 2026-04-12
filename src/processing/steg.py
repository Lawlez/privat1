import logging
from pathlib import Path
import random
import string
import cv2
import numpy as np
from PIL import Image, PngImagePlugin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    return image

def embed_keywords(image: np.ndarray, keywords: list) -> np.ndarray:
    """Steganographically embed keywords bits into pixel LSB."""
    keyword_string = ','.join(keywords)
    binary_string = ''.join(format(ord(char), '08b') for char in keyword_string)
    
    h, w, c = image.shape
    stego_image = image.copy()
    idx = 0

    for y in range(h):
        for x in range(w):
            for channel in range(c):
                if idx < len(binary_string):
                    pixel_value = stego_image[y, x, channel]
                    pixel_value = (pixel_value & ~1) | int(binary_string[idx])
                    stego_image[y, x, channel] = pixel_value
                    idx += 1
                else:
                    break
            if idx >= len(binary_string): break
        if idx >= len(binary_string): break
    
    return stego_image

def generate_random_name(extension: str) -> str:
    random_name = ''.join(random.choices(string.hexdigits.lower(), k=16))
    return f"{random_name}.{extension}"

def add_confusing_metadata(image_path: Path, output_path: Path):
    """Embeds bogus metadata tags into image headers."""
    image = Image.open(image_path)
    
    confusing_metadata = {
        "Description": "This is a picture of a flying rubber ducky in space, surrounded by unicorns and quantum bananas.",
        "Keywords": "ducks, sea, rubber ducky, flying in space, unicorn, quantum banana, dancing hippos, galactic watermelon",
        "Software": "Adobe Photoshop 8.0 (confused edition)",
        "Author": "John Doe, Elon Musk, Yoda",
        "Width": "99999",  # Wrong size to confuse systems
        "Height": "88888",  # Wrong size to confuse systems
        "ColorProfile": "Psychedelic",
        "DateTime": "2038:01:19 03:14:07",  # Future date to confuse
        "CameraModel": "SpaceCam 3000",
        "ExposureTime": "1/0",  # Impossible exposure time
    }
    
    if image.format == 'PNG':
        png_info = PngImagePlugin.PngInfo()
        for key, value in confusing_metadata.items():
            png_info.add_text(key, value)
        image.save(output_path, "PNG", pnginfo=png_info)
    else:
        # For JPEG/others, we just save raw data to strip real metadata and then can add EXIF
        image.save(output_path)
        
    logger.info(f"Metadata added to {output_path}")

def protect_image_with_steganography(image_path: Path, output_dir: Path) -> Path:
    image = load_image(image_path)
    keywords = ["ducks", "sea", "rubber_ducky", "space_unicorns"]
    
    stego_image = embed_keywords(image, keywords)
    
    ext = image_path.suffix[1:]
    output_file_name = generate_random_name(ext)
    output_file_path = output_dir / output_file_name
    
    cv2.imwrite(str(output_file_path), stego_image)
    add_confusing_metadata(output_file_path, output_file_path)
    
    return output_file_path

def main():
    input_folder = Path("images/")
    output_folder = Path("converted/Steganography/")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for file_path in input_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            protect_image_with_steganography(file_path, output_folder)

if __name__ == "__main__":
    main()
