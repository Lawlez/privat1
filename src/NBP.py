import cv2
import numpy as np
import os
import random
import string
from PIL import Image

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

# Generate a random 8-byte hexadecimal name
def generate_random_name(extension):
    random_name = ''.join(random.choices(string.hexdigits.lower(), k=16))
    return f"{random_name}.{extension}"

# Combine methods to protect identity in an image
def protect_image(image_path, output_path):
    # Load the image
    image = load_image(image_path)
    
    # Apply series of transformations
    image = apply_blur(image)
    image = add_noise(image)
    image = pixelate(image)
    
    # Get the file extension
    _, ext = os.path.splitext(image_path)
    ext = ext[1:]  # remove the leading '.'

    # Generate a random name for the output image
    output_file_name = generate_random_name(ext)
    output_file_path = os.path.join(output_path, output_file_name)
    
    # Save the modified image
    cv2.imwrite(output_file_path, image)
    print(f"Protected image saved to {output_file_path}")
    
    return output_file_path

# Remove metadata using PIL
def remove_metadata(image_path, output_path):
    image = Image.open(image_path)
    data = list(image.getdata())
    image_no_metadata = Image.new(image.mode, image.size)
    image_no_metadata.putdata(data)
    image_no_metadata.save(output_path)
    print(f"Metadata removed from image and saved to {output_path}")

if __name__ == "__main__":
    input_folder = "/Users/lwlx/PROJECTS/privat1/images/"  # Replace with your image folder path
    output_folder = "/Users/lwlx/PROJECTS/privat1/converted/PixelShift/"  # Replace with your desired output folder path
    
    # Make sure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, file_name)
        
        # Only process files with valid image extensions
        if os.path.isfile(input_image_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            protected_image_path = protect_image(input_image_path, output_folder)

            # Remove metadata from the protected image
            metadata_removed_path = os.path.join(output_folder, f"no_metadata_{os.path.basename(protected_image_path)}")
            remove_metadata(protected_image_path, metadata_removed_path)
