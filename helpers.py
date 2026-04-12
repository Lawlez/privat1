import logging
from typing import Tuple, Optional
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import losses
from art.estimators.classification import TensorFlowV2Classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global cache for the model to avoid redundant loads
_MODEL_CACHE = {}

def create_tf_hub_classifier(
    handle: str = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5",
    input_shape: Tuple[int, int, int] = (224, 224, 3)
) -> TensorFlowV2Classifier:
    """
    Loads a classification model from TensorFlow Hub, wraps it in a Keras Model, 
    and initializes an ART classifier.
    
    Optimized for macOS with tensorflow-metal if available.
    """
    if handle in _MODEL_CACHE:
        logger.debug(f"Using cached model for {handle}")
        return _MODEL_CACHE[handle]

    logger.info(f"Loading model from {handle}...")
    
    try:
        # Load the base model
        base_model = hub.load(handle)
        
        # Define a Keras Model wrapper
        class HubModel(tf.keras.Model):
            def call(self, inputs):
                return base_model(inputs)

        model = HubModel()

        # Determine number of output classes dynamically
        dummy_input = tf.random.uniform((1, *input_shape), minval=0.0, maxval=1.0)
        output = model(dummy_input)
        num_classes = output.shape[-1]

        # Categorical cross-entropy for ImageNet-style models
        loss_fn = losses.CategoricalCrossentropy(from_logits=False)

        # Wrap in an ART classifier
        art_classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=num_classes,
            input_shape=input_shape,
            loss_object=loss_fn,
            clip_values=(0.0, 1.0)
        )
        
        _MODEL_CACHE[handle] = art_classifier
        return art_classifier

    except Exception as e:
        logger.error(f"Failed to load model from {handle}: {e}")
        raise
