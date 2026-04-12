import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from helpers import create_tf_hub_classifier
from tensorflow.keras.preprocessing import image

# Load pre-trained MobileNetV2 model wrapped in an ART classifier
classifier = create_tf_hub_classifier()

# Load and preprocess an image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

img_path = "img4.jpg"  # Change to your image path
x_test = load_image(img_path)

# Apply a small rotation (5 degrees)
def spatial_transform(img, angle=5):
    img_np = np.squeeze(img)  # Remove batch dimension
    h, w = img_np.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # Rotation matrix
    transformed = cv2.warpAffine(img_np, matrix, (w, h))
    return np.expand_dims(transformed, axis=0)  # Restore batch dimension

x_adv = spatial_transform(x_test, angle=5)

# Display original vs adversarial
plt.subplot(1, 2, 1)
plt.imshow(x_test[0])
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(x_adv[0])
plt.title("Spatially Transformed Attack")

plt.show()
