import logging
import threading
import requests
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import losses
from art.estimators.classification import TensorFlowV2Classifier

import re

# --- Colored Logging Setup ---
class ColoredFormatter(logging.Formatter):
    """Custom formatter for premium colored console output with keyword highlighting."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # White on Red
    }
    HIGHLIGHT = '\033[91m'      # Bright Red
    RESET = '\033[0m'
    
    # Keywords to highlight in RED
    TARGETS = [r'Applying', r'PGD', r'FGM', r'Pixelation', r'Pixel Shift', r'Pattern Mask', r'Perspective']

    def format(self, record):
        msg = super().format(record)
        # Apply standard level coloring
        color = self.COLORS.get(record.levelname, self.RESET)
        msg = msg.replace(record.levelname, f"{color}{record.levelname}{self.RESET}")
        
        # Apply keyword highlighting
        for target in self.TARGETS:
            msg = re.sub(f'({target})', f'{self.HIGHLIGHT}\\1{self.RESET}', msg)
        
        return msg

def box_msg(msg: str):
    """Returns a string enclosed in an ASCII box for status headers."""
    border = '─' * (len(msg) + 2)
    return f"\n┌{border}┐\n│ {msg} │\n└{border}┘"

def setup_logger(name: str):
    """Configures a logger with colored output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColoredFormatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

logger = setup_logger(__name__)

# Constants
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
LABELS_FILE = Path("imagenet_labels.txt")

def ensure_labels():
    """Downloads and caches ImageNet labels."""
    if not LABELS_FILE.exists():
        logger.info("Downloading ImageNet labels...")
        try:
            response = requests.get(LABELS_URL, timeout=10)
            response.raise_for_status()
            LABELS_FILE.write_text(response.text)
        except Exception as e:
            logger.error(f"Failed to download labels: {e}")
            raise
    return [line.strip() for line in LABELS_FILE.read_text().splitlines()]

# Global cache for the model to avoid redundant loads
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()

class RobustTensorFlowV2Classifier(TensorFlowV2Classifier):
    """
    Sub-classed ART classifier to fix the persistent GradientTape warning 
    and improve stability in Keras 3 environments.
    """
    def loss_gradient(self, x, y, training=False, **kwargs):
        """
        Efficient gradient calculation using a non-persistent tape,
        calculating gradients outside the context to satisfy Keras 3 efficiency requirements.
        """
        x_tensor = tf.convert_to_tensor(x)
        y_tensor = tf.convert_to_tensor(y)
        
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predictions = self._model(x_tensor, training=training)
            loss = self._loss_object(y_tensor, predictions)
            
        gradients = tape.gradient(loss, x_tensor)
        return gradients.numpy()

def create_tf_hub_classifier(
    handle: str = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5",
    input_shape: tuple[int, int, int] = (224, 224, 3)
) -> RobustTensorFlowV2Classifier:
    """
    Loads a classification model from TensorFlow Hub, wraps it in a Keras Model, 
    and initializes an ART classifier.
    """
    with _MODEL_LOCK:
        if handle in _MODEL_CACHE:
            logger.debug(f"Using cached model for {handle}")
            return _MODEL_CACHE[handle]

        logger.info(f"Loading model from {handle}...")
        try:
            class HubLayer(tf.keras.layers.Layer):
                def __init__(self, handle, **kwargs):
                    super().__init__(**kwargs)
                    self.hub_module = hub.load(handle)
                    # Convert to concrete lists immediately to avoid iterator issues in multi-threading
                    self._trainable_vars = list(self.hub_module.trainable_variables)
                    self._all_vars = list(self.hub_module.variables)
                    
                def call(self, inputs):
                    return self.hub_module(inputs)

                @property
                def trainable_variables(self):
                    return self._trainable_vars
                
                @property
                def variables(self):
                    return self._all_vars

            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(shape=input_shape),
                HubLayer(handle)
            ])

            # Determine number of output classes dynamically
            dummy_input = tf.random.uniform((1, *input_shape), minval=0.0, maxval=1.0)
            output = model(dummy_input)
            num_classes = output.shape[-1]

            loss_fn = losses.CategoricalCrossentropy(from_logits=True)

            # Wrap in our robust custom classifier
            art_classifier = RobustTensorFlowV2Classifier(
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
