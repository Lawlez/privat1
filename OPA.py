import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from helpers import create_tf_hub_classifier
from art.attacks.evasion import PixelAttack
from tensorflow.keras.preprocessing import image

classifier = create_tf_hub_classifier()

# Load and preprocess an image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

img_path = "img4.jpg"  # Change to your image path
x_test = load_image(img_path)

# Generate adversarial example using PixelAttack
attack = PixelAttack(classifier, th=1, es=1)  # `th=1` = one pixel changed
x_adv = attack.generate(x=x_test)

# Display original vs adversarial
plt.subplot(1, 2, 1)
plt.imshow(x_test[0])
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(x_adv[0])
plt.title("One-Pixel Attack")

plt.show()
