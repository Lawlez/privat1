import os
import cv2
import numpy as np
import random
import string
from PIL import Image, PngImagePlugin, JpegImagePlugin

# Load image using OpenCV
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    return image

# Steganography to embed nonsensical keywords into the image
def embed_keywords(image, keywords, intensity=5):
    # Flatten the keywords into a single string
    keyword_string = ','.join(keywords)
    
    # Convert the string to binary representation
    binary_string = ''.join(format(ord(char), '08b') for char in keyword_string)
    
    # Get image dimensions
    h, w, c = image.shape
    
    # Create a copy of the image to embed the data
    stego_image = image.copy()
    idx = 0

    # Embed binary data into image pixels
    for y in range(h):
        for x in range(w):
            for channel in range(c):
                if idx < len(binary_string):
                    # Modify the pixel value by embedding one bit of the binary string
                    pixel_value = stego_image[y, x, channel]
                    pixel_value = (pixel_value & ~1) | int(binary_string[idx])  # Set LSB to the bit value
                    stego_image[y, x, channel] = pixel_value
                    idx += 1
                else:
                    break
            if idx >= len(binary_string):
                break
        if idx >= len(binary_string):
            break
    
    return stego_image

# Generate a random 8-byte hexadecimal name
def generate_random_name(extension):
    random_name = ''.join(random.choices(string.hexdigits.lower(), k=16))
    return f"{random_name}.{extension}"

# Embed confusing metadata into the image
def add_confusing_metadata(image_path, output_path):
    image = Image.open(image_path)
    metadata = image.info

    # Create confusing metadata
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
    
    # Update or add the confusing metadata
    for key, value in confusing_metadata.items():
        metadata[key] = value
    
    # Save image with metadata
    if image.format == 'JPEG':
        exif = image.getexif()
        for key, value in confusing_metadata.items():
            exif[270] = value  # Tag 270 is commonly used for image description
        image.save(output_path, "JPEG", exif=exif)
    elif image.format == 'PNG':
        png_info = PngImagePlugin.PngInfo()
        for key, value in confusing_metadata.items():
            png_info.add_text(key, value)
        image.save(output_path, "PNG", pnginfo=png_info)
    else:
        image.save(output_path)

    print(f"Metadata added to image and saved to {output_path}")
    return output_path

# Combine methods to protect identity in an image
def protect_image_with_steganography(image_path, output_path):
    # Load the image
    image = load_image(image_path)
    
    # Keywords to embed for confusion
    keywords = ["ducks", "sea", "rubber ducky", "flying in space", "unicorn", "quantum banana"]
    
    # Apply steganography to embed nonsensical keywords
    image = embed_keywords(image, keywords)
    
    # Get the file extension
    _, ext = os.path.splitext(image_path)
    ext = ext[1:]  # remove the leading '.'

    # Generate a random name for the output image
    output_file_name = generate_random_name(ext)
    output_file_path = os.path.join(output_path, output_file_name)
    
    # Save the modified image
    cv2.imwrite(output_file_path, image)
    print(f"Protected image with steganography saved to {output_file_path}")
    
    # Add confusing metadata to the image
    add_confusing_metadata(output_file_path, output_file_path)
    
    return output_file_path

if __name__ == "__main__":
    input_folder = "/Users/lwlx/PROJECTS/privat1/images/"  # Replace with your image folder path
    output_folder = "/Users/lwlx/PROJECTS/privat1/converted/Steganography/"  # Replace with your desired output folder path
    
    # Make sure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, file_name)
        
        # Only process files with valid image extensions
        if os.path.isfile(input_image_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            protect_image_with_steganography(input_image_path, output_folder)
