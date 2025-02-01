import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load ImageNet labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
LABELS_FILE = "imagenet_labels.txt"

if not os.path.exists(LABELS_FILE):
    print("Downloading ImageNet labels...")
    import requests
    response = requests.get(LABELS_URL)
    with open(LABELS_FILE, "w") as f:
        f.write(response.text)

# Read labels
with open(LABELS_FILE, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load the MobileNet model from TF Hub
print("Loading MobileNetV2 model...")
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")

def classify_image(image_path, forbidden_classes):
    """
    Classifies an image and checks if any of the detected labels match the forbidden classes.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Failed to load image: {image_path}")
        return
    
    # Resize and normalize
    img_resized = cv2.resize(image, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Run classification
    predictions = model(img_expanded).numpy()
    top5_indices = np.argsort(predictions[0])[-10:][::-1]  # Get top 5 predictions

    detected_labels = [(labels[idx], idx, predictions[0][idx]) for idx in top5_indices]

    # Print results
    print(f"\n🖼️ Image: {image_path}")
    for name, idx, prob in detected_labels:
        print(f"  - {name} (ID {idx}) -> Confidence: {prob:.4f}")

    # Check if any detected label is in the forbidden list
    detected_names = {name.lower() for name, _, _ in detected_labels}
    forbidden_names = {cls.lower() for cls in forbidden_classes}

    if detected_names & forbidden_names:
        print("❌ Forbidden object detected!")
    else:
        print("✅ Success: No forbidden objects detected.")

def process_directory(directory_path, forbidden_classes):
    """
    Processes all images in the given directory.
    """
    if not os.path.exists(directory_path):
        print("⚠️ Directory does not exist!")
        return

    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    if not image_files:
        print("⚠️ No valid image files found in the directory!")
        return

    print(f"\n📂 Processing {len(image_files)} images in '{directory_path}'...\n")
    for image_file in image_files:
        classify_image(os.path.join(directory_path, image_file), forbidden_classes)

# Set your directory and forbidden classes here
IMAGE_DIRECTORY = "/Users/lwlx/PROJECTS/privat1/images"
FORBIDDEN_CLASSES = ["keeshond", "church", "suit", "mask"]  # Add classes you want to avoid

process_directory(IMAGE_DIRECTORY, FORBIDDEN_CLASSES)
