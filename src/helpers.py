import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import losses

from art.estimators.classification import TensorFlowV2Classifier

def create_tf_hub_classifier():
    """
    Loads a MobileNetV2 classification model (ImageNet) directly as a SavedModel,
    wraps it in a custom tf.keras.Model subclass, then wraps *that* in ART.
    """
    # This loads a SavedModel from TF Hub:
    #   (there are many other model variants; pick whichever you prefer)
    base_model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")
    
    # We'll define a small Keras Model that calls 'base_model' in its forward pass.
    class HubModel(tf.keras.Model):
        def call(self, inputs):
            # base_model expects inputs in shape (batch, 224, 224, 3) and range [0,1]
            return base_model(inputs)
    
    model = HubModel()

    # We'll use CategoricalCrossentropy for multi-class probabilities (ImageNet).
    loss_fn = losses.CategoricalCrossentropy(from_logits=False)

    # Wrap the model in an ART TensorFlowV2Classifier
    # MobileNet usually outputs 1001 classes for ImageNet, sometimes 1000. 
    # Check the exact shape if you want to be precise.
    art_classifier = TensorFlowV2Classifier(
        model=model,
        nb_classes=1001,
        input_shape=(224, 224, 3),
        loss_object=loss_fn
    )
    return art_classifier