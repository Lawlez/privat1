import os
import cv2
import numpy as np
import random
import string
from PIL import Image, PngImagePlugin
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Load image using OpenCV
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    return image

# Apply adversarial noise to the image using ART without TensorFlow
def apply_adversarial_noise(image, epsilon=0.0035):  # Lowered epsilon to reduce visibility of noise
    # Flatten the image to 2D for the model
    h, w, c = image.shape
    image_flattened = image.astype(np.float32).reshape(1, -1) / 255.0

    # Create a simple scikit-learn model for demonstration purposes
    X_train, y_train = make_classification(n_samples=100, n_features=h * w * c, n_classes=2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    classifier = SklearnClassifier(model=model)

    # Create the adversarial attack using Fast Gradient Method (FGM)
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    adversarial_image_flattened = attack.generate(x=image_flattened)

    # Reshape the adversarial image back to its original shape
    adversarial_image = adversarial_image_flattened.reshape(h, w, c) * 255.0
    adversarial_image = adversarial_image.astype(np.uint8)

    # Apply slight Gaussian blur to reduce visible noise artifacts
    adversarial_image = cv2.GaussianBlur(adversarial_image, (3, 3), 0)

    return adversarial_image

# Apply pixel shift to distort image in a subtle but AI-confusing way
def apply_pixel_shift(image, shift_amount=2):
    shifted_image = image.copy()
    h, w, c = shifted_image.shape
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if x + shift_amount < w and y + shift_amount < h:
                shifted_image[y, x] = image[(y + shift_amount) % h, (x + shift_amount) % w]
    return shifted_image

# Apply a pixel pattern mask to subtly alter pixel values
def apply_pixel_pattern_mask(image, pattern_size=4, opacity=0.2):
    masked_image = image.copy()
    h, w, c = masked_image.shape
    pattern = (np.random.randint(0, 2, (pattern_size, pattern_size, c), dtype='uint8') * 50).astype(np.float32)
    for y in range(0, h, pattern_size):
        for x in range(0, w, pattern_size):
            region = masked_image[y:y+pattern_size, x:x+pattern_size].astype(np.float32)
            blended_region = cv2.addWeighted(region, 1 - opacity, pattern[:min(h-y, pattern_size), :min(w-x, pattern_size)], opacity, 0)
            masked_image[y:y+pattern_size, x:x+pattern_size] = blended_region.astype(np.uint8)
    return masked_image

# Apply Gaussian Blur to distort facial features
def apply_blur(image, kernel_size=(1, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Add random noise to the image to confuse recognition models
def apply_noise(image, noise_level=10):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noise = np.random.randint(-noise_level, noise_level, (h, w, c), dtype='int16')
    noisy_image = cv2.add(noisy_image, noise, dtype=cv2.CV_8U)
    return noisy_image

# Apply a pixelation effect
def apply_pixelation(image, pixel_size=2):
    height, width = image.shape[:2]
    temp = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

# Modify compression to further distort image characteristics
def apply_compression(image, quality=85):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encimg, 1)
    return compressed_image

# Generate a random 8-byte hexadecimal name
def generate_random_name(extension):
    random_name = ''.join(random.choices(string.hexdigits.lower(), k=16))
    return f"{random_name}.{extension}"


# Steganography to embed nonsensical keywords into the image and add confusing metadata
def embed_keywords_and_metadata(image, keywords, metadata):
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
    
    # Convert OpenCV image to PIL format to add metadata
    pil_image = Image.fromarray(cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB))
    png_info = PngImagePlugin.PngInfo()
    
    # Add confusing metadata
    for key, value in metadata.items():
        png_info.add_text(key, value)
    
    # Save image with metadata
    output_path = "output_with_metadata.png"
    pil_image.save(output_path, "PNG", pnginfo=png_info)
    print(f"Metadata added to image and saved to {output_path}")
    
    return output_path

# Combine methods to protect identity in an image
def protect_image(image_path, output_path):
    # Load the image
    image = load_image(image_path)
    
    # Apply series of transformations
    image = apply_adversarial_noise(image)
    image = apply_pixel_shift(image)
    image = apply_pixel_pattern_mask(image)
    image = apply_compression(image)
    image = apply_blur(image)
    image = apply_noise(image)
    image = apply_pixelation(image)
    
    # Keywords and metadata for steganography
    keywords = ["ducks", "sea", "rubber ducky", "flying in space", "unicorn", "quantum banana"]
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
    
    # Embed keywords and metadata
    embed_keywords_and_metadata(image, keywords, confusing_metadata)
    
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
