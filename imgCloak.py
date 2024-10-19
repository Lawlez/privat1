import cv2
import numpy as np
from PIL import Image, ImageFilter
import random

# Load image using OpenCV
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    return image

# Apply Gaussian Blur to distort facial features
def apply_blur(image, kernel_size=(1, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Add random noise to the image to confuse recognition models
def add_noise(image, noise_level=8):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noise = np.random.randint(-noise_level, noise_level, (h, w, c), dtype='int16')
    noisy_image = cv2.add(noisy_image, noise, dtype=cv2.CV_8U)
    return noisy_image

# Apply a pixelation effect
def pixelate(image, pixel_size=1):
    height, width = image.shape[:2]
    temp = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

# Combine methods to protect identity in an image
def protect_image(image_path, output_path):
    # Load the image
    image = load_image(image_path)
    
    # Apply series of transformations
    image = apply_blur(image)
    image = add_noise(image)
    image = pixelate(image)
    
    # Save the modified image
    cv2.imwrite(output_path, image)
    print(f"Protected image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = "/Users/lwlx/PROJECTS/privat1/images/th3.jpeg"  # Replace with your image path
    output_image_path = "/Users/lwlx/PROJECTS/privat1/converted/PixelShift/output_protected.jpg"  # Replace with your desired output path
    protect_image(input_image_path, output_image_path)
