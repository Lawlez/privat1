#!/usr/bin/env python3

import os
import cv2
import numpy as np
import random
import string
from PIL import Image, PngImagePlugin

from PGD import apply_pgd_with_upsized_delta
from FGM import apply_fgm_with_upsized_delta
from carlini_wagner import apply_cwl2_with_upsized_delta

def load_image(image_path):
    """Loads image via OpenCV."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    return image

def generate_random_name(extension):
    """Generates a random 16-hex-digit filename."""
    random_name = ''.join(random.choices(string.hexdigits.lower(), k=16))
    return f"{random_name}.{extension}"

def remove_metadata(image_path, output_path):
    """Strips metadata by re-saving pixel data with default PIL settings."""
    image = Image.open(image_path)
    data = list(image.getdata())
    image_no_metadata = Image.new(image.mode, image.size)
    image_no_metadata.putdata(data)
    image_no_metadata.save(output_path)
    print(f"Metadata removed and saved to {output_path}")

############################################
#         IMAGE DISTORTION METHODS         #
############################################

def apply_pixel_shift(image, shift_amount=8):
    print("Applying Pixel Shift.")
    shifted_image = image.copy()
    h, w, c = shifted_image.shape
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if (x + shift_amount) < w and (y + shift_amount) < h:
                shifted_image[y, x] = image[(y + shift_amount) % h, (x + shift_amount) % w]
    return shifted_image

def apply_pixel_pattern_mask(image, pattern_size=5, opacity=0.25):
    print("Applying Pixel Pattern Mask.")
    masked_image = image.copy()
    h, w, c = masked_image.shape
    pattern = (np.random.randint(0, 2, (pattern_size, pattern_size, c), dtype='uint8') * 70).astype(np.float32)
    for y in range(0, h, pattern_size):
        for x in range(0, w, pattern_size):
            region = masked_image[y:y+pattern_size, x:x+pattern_size].astype(np.float32)
            blend = cv2.addWeighted(region, 1 - opacity, pattern[:region.shape[0], :region.shape[1]], opacity, 0)
            masked_image[y:y+pattern_size, x:x+pattern_size] = blend.astype(np.uint8)
    return masked_image

def apply_blur(image, kernel_size=(3, 3)):
    print("Applying Blur.")
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_noise(image, noise_level=20):
    print("Applying Random Noise.")
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noise = np.random.randint(-noise_level, noise_level, (h, w, c), dtype='int16')
    noisy_image = cv2.add(noisy_image, noise, dtype=cv2.CV_8U)
    return noisy_image

def apply_pixelation(image, pixel_size=4):
    print("Applying Pixelation.")
    height, width = image.shape[:2]
    temp = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def apply_compression(image, quality=75):
    print("Applying JPEG compression.")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encimg, 1)
    return compressed_image

def apply_random_perspective_transform(image):
    print("Applying Random Perspective Transform.")
    h, w = image.shape[:2]
    margin = 0.08
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])

    def rand_pt(x, y):
        return [
            x + random.randint(-int(w*margin), int(w*margin)),
            y + random.randint(-int(h*margin), int(h*margin))
        ]

    pts2 = np.float32([rand_pt(0,0),
                       rand_pt(w,0),
                       rand_pt(0,h),
                       rand_pt(w,h)])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped

############################################
#            METADATA & STEGO              #
############################################

def embed_keywords_and_metadata(image, keywords, metadata):
    print("Embedding stego keywords & changing metadata.")
    keyword_string = ','.join(keywords)
    binary_string = ''.join(format(ord(char), '08b') for char in keyword_string)

    h, w, c = image.shape
    stego_image = image.copy()
    idx = 0

    # Hide bits in LSB of image
    for y in range(h):
        for x in range(w):
            for channel in range(c):
                if idx < len(binary_string):
                    pixel_value = stego_image[y, x, channel]
                    # Safe bit-twiddle for uint8
                    pixel_value = (pixel_value & 0xFE) | int(binary_string[idx])
                    stego_image[y, x, channel] = pixel_value
                    idx += 1
                else:
                    break
            if idx >= len(binary_string):
                break
        if idx >= len(binary_string):
            break

    pil_image = Image.fromarray(cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB))
    png_info = PngImagePlugin.PngInfo()

    for key, value in metadata.items():
        png_info.add_text(key, value)

    out_path = "output_with_metadata.png"
    pil_image.save(out_path, "PNG", pnginfo=png_info)
    print(f"Metadata-laden image saved to {out_path}")
    return out_path

def embed_resized_images(original_image, assets_path):
    print("Embedding small training images in random regions.")
    h, w, c = original_image.shape
    resized_images = []

    for file_name in os.listdir(assets_path):
        image_path = os.path.join(assets_path, file_name)
        if (os.path.isfile(image_path) and 
            file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))):
            asset_image = load_image(image_path)
            resized_image = cv2.resize(asset_image, (28, 28))
            resized_images.append(resized_image)

    for resized_image in resized_images:
        for _ in range(9):
            y_offset = random.randint(0, h - 28)
            x_offset = random.randint(0, w - 28)

            roi = original_image[y_offset:y_offset+28, x_offset:x_offset+28]
            resized_image_float = resized_image.astype(np.float32) / 255.0
            roi_float = roi.astype(np.float32) / 255.0

            # 'Screen' blend + partial opacity
            blended_region = 1 - (1 - roi_float) * (1 - resized_image_float)
            blended_region = (blended_region * 0.5 + roi_float * 0.5)
            blended_region = np.clip(blended_region * 255.0, 0, 255).astype(np.uint8)

            original_image[y_offset:y_offset+28, x_offset:x_offset+28] = blended_region

    return original_image

############################################
#            MAIN PROTECTION FLOW          #
############################################

def protect_image(image_path, output_path):
    """Chained transformations + metadata confusion + stego + TF-based adversarial attacks."""
    image = load_image(image_path)
    print(f"Loaded image from {image_path}")

    # Distort the image in multiple ways
    image = apply_blur(image, kernel_size=(3, 3))
    image = apply_pixel_shift(image, shift_amount=6)
    image = apply_pixel_pattern_mask(image, pattern_size=4, opacity=0.3)
    image = apply_random_perspective_transform(image)
    image = apply_noise(image, noise_level=20)
    image = apply_pixelation(image, pixel_size=3)
    image = apply_compression(image, quality=80)

    # Multi-step adversarial approach (now all TensorFlow-based)
    image = apply_pgd_with_upsized_delta(image, eps=0.09, eps_step=0.1, max_iter=30)
    image = apply_fgm_with_upsized_delta(image, epsilon=0.06, eps_steps=0.1, target_label=0)
    image = apply_cwl2_with_upsized_delta(image, confidence=3.0, max_iter=25)

    # Optionally embed random assets
    #image = embed_resized_images(image, "./assets")

    # Add dummy metadata and stego keywords
    keywords = ["ducks", "sea", "rubber ducky", "flying in space", "unicorn", "quantum banana"]
    confusing_metadata = {
        "Description": "This is an image of quantum bananas with interstellar rubber ducks. Confusion intensifies.",
        "Keywords": "ducks, space bananas, dancing hippos, quantum mechanics, asparagus rocket",
        "Software": "Adobe GlobiShop 99.0 (ultimate confusion edition)",
        "Author": "Elon Musk, John Doe, Chewbacca",
        "Width": "99999",
        "Height": "88888",
        "ColorProfile": "Psychedelic Dream Sequence",
        "DateTime": "4:20:13:37 07:07:07",
        "CameraModel": "SpaceCam 9001",
        "ExposureTime": "1/∞",
    }
    embed_keywords_and_metadata(image, keywords, confusing_metadata)

    # Save final obfuscated image
    _, ext = os.path.splitext(image_path)
    ext = ext[1:] 
    output_file_name = f'{generate_random_name(ext)}'
    output_file_path = os.path.join(output_path, output_file_name)
    cv2.imwrite(output_file_path, image)
    print(f"Protected image saved to {output_file_path}")

    return output_file_path

############################################
#               ENTRY POINT                #
############################################

if __name__ == "__main__":
    input_folder = "/Users/lwlx/PROJECTS/privat1/images/" 
    output_folder = "/Users/lwlx/PROJECTS/privat1/converted/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, file_name)
        if (os.path.isfile(input_image_path) and 
            file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))):
            protected_image_path = protect_image(input_image_path, output_folder)

            # Remove metadata from final product
            meta_removed_path = os.path.join(
                output_folder, f"no_metadata_{os.path.basename(protected_image_path)}"
            )
            remove_metadata(protected_image_path, meta_removed_path)
