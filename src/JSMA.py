import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

def preprocess_image(image_path, target_size=(300, 300)):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, target_size)  # Resize to the specified input size for the model
    return img

def image_to_array(image):
    # Convert PIL image to numpy array
    img_array = np.asarray(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def jsma_attack(model, x, target, theta=1.0, gamma=0.2, max_iterations=500, perturbation_step=0.01):
    
    
    x_var = tf.convert_to_tensor(x, dtype=tf.float32)
    target = tf.constant(target, dtype=tf.float32)
    
    # Calculate gradients
    with tf.GradientTape() as tape:
        tape.watch(x_var)
        y_pred = model(x_var)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=y_pred)
    gradients = tape.gradient(loss, x_var)
    
    # Perform the iterative attack
    for _ in range(min(max_iterations, int(gamma * np.prod(x.shape)))):
        grad_values = gradients
        
        # Calculate Jacobian Saliency Map
        # Pick the pixel with the highest gradient to modify
        indices = np.unravel_index(np.argmax(grad_values), grad_values.shape)
        x = tf.tensor_scatter_nd_add(x_var, [indices], [perturbation_step * theta])
        x = tf.clip_by_value(x, 0, 1)
        
    return x

def main(image_path, model):
    # Preprocess the image
    original_image = preprocess_image(image_path, target_size=(300, 300))
    
    # Convert the image to a numpy array
    input_array = image_to_array(original_image)
    
    # Define the target class as a one-hot vector (for demonstration purposes, assuming a binary target)
    target_class = np.zeros((1, 1000))  # Adjust target class size to match the model's output
    target_class[0, 3] = 1  # Let's assume we want to target class 3
    
    # Run the JSMA
    adversarial_image = jsma_attack(model, input_array, target_class, theta=1.5, gamma=0.3, max_iterations=1000, perturbation_step=0.02)
    
    # Convert the adversarial image back to a format suitable for saving/display
    adversarial_image = adversarial_image.squeeze() * 255
    adversarial_image = adversarial_image.astype(np.uint8)
    adversarial_pil = Image.fromarray(adversarial_image)
    adversarial_pil.save("adversarial_example.png")
    adversarial_pil.show()

if __name__ == "__main__":
    # Load a pre-trained model (for example, MobileNetV2 from TensorFlow)
    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    model = tf.keras.Sequential([
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1000, activation='softmax')
    ])
    
    # Path to the input image
    image_path = "input_image.jpg"
    
    main(image_path, model)
