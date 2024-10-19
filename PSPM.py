import os
import cv2
import numpy as np
import random
import string
from PIL import Image
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

# Apply adversarial noise to the image using ART
def apply_adversarial_noise(image, epsilon=0.0035):
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

    return adversarial_image

# Apply pixel shift to distort image in a subtle but AI-confusing way
def apply_pixel_shift(image, shift_amount=4):
    shifted_image = image.copy()
    h, w, c = shifted_image.shape
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if x + shift_amount < w and y + shift_amount < h:
                shifted_image[y, x] = image[(y + shift_amount) % h, (x + shift_amount) % w]
    return shifted_image

# Apply a pixel pattern mask to subtly alter pixel values
def apply_pixel_pattern_mask(image, pattern_size=4):
    masked_image = image.copy()
    h, w, c = masked_image.shape
    pattern = np.random.randint(0, 2, (pattern_size, pattern_size, c), dtype='uint8') * 20
    for y in range(0, h, pattern_size):
        for x in range(0, w, pattern_size):
            masked_image[y:y+pattern_size, x:x+pattern_size] = cv2.add(masked_image[y:y+pattern_size, x:x+pattern_size], pattern[:min(h-y, pattern_size), :min(w-x, pattern_size)])
    return masked_image

# Modify compression to further distort image characteristics
def apply_compression(image, quality=95):
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
    image = apply_adversarial_noise(image)
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
