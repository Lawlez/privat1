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

# Apply pixel shift to distort image in a subtle but AI-confusing way
def apply_pixel_shift(image, shift_amount=1):
    shifted_image = image.copy()
    h, w, c = shifted_image.shape
    for y in range(0, h, 1):
        for x in range(0, w, 2):
            if x + shift_amount < w and y + shift_amount < h:
                shifted_image[y, x] = image[(y + shift_amount) % h, (x + shift_amount) % w]
    return shifted_image

# Apply a pixel pattern mask to subtly alter pixel values
def apply_pixel_pattern_mask(image, pattern_size=4):
    masked_image = image.copy()
    h, w, c = masked_image.shape
    pattern = np.random.randint(0, 2, (pattern_size, pattern_size, c), dtype='uint8') * 50
    for y in range(0, h, pattern_size):
        for x in range(0, w, pattern_size):
            masked_image[y:y+pattern_size, x:x+pattern_size] = cv2.add(masked_image[y:y+pattern_size, x:x+pattern_size], pattern[:min(h-y, pattern_size), :min(w-x, pattern_size)])
    return masked_image

# Modify compression to further distort image characteristics
def apply_compression(image, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encimg, 1)
    return compressed_image

# Generate a random 8-byte hexadecimal name
def generate_random_name(extension):
    random_name = ''.join(random.choices(string.hexdigits.lower(), k=16))
    return f"{random_name}.{extension}"

# Combine methods to protect identity in an image
def protect_image(image_path, output_path):
    # Load the image
    image = load_image(image_path)
    
    # Apply series of transformations
    image = apply_pixel_shift(image)
    image = apply_pixel_pattern_mask(image)
    image = apply_compression(image)
    
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
