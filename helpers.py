import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import losses
from art.estimators.classification import TensorFlowV2Classifier

def create_tf_hub_classifier():
    """
    Loads a MobileNetV2 classification model (ImageNet) directly as a SavedModel,
    wraps it in a custom tf.keras.Model subclass, then wraps *that* in ART.
    """
    # Load the MobileNetV2 model from TensorFlow Hub
    base_model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")
    
    # Define a Keras Model that wraps the TF Hub model
    class HubModel(tf.keras.Model):
        def call(self, inputs):
            return base_model(inputs)  # Output: (batch_size, num_classes)

    model = HubModel()

    # Determine number of output classes dynamically
    dummy_input = tf.random.uniform((1, 224, 224, 3), minval=0.0, maxval=1.0)
    output = model(dummy_input)
    num_classes = output.shape[-1]  # Should be 1000 or 1001

    # Use categorical cross-entropy for multi-class classification
    loss_fn = losses.CategoricalCrossentropy(from_logits=False)

    # Wrap in an ART classifier
    art_classifier = TensorFlowV2Classifier(
        model=model,
        nb_classes=num_classes,  # Dynamically determined
        input_shape=(224, 224, 3),
        loss_object=loss_fn,
        clip_values=(0.0, 1.0)  # VERY IMPORTANT for PGD attacks
    )
    
    return art_classifier
