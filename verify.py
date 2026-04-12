import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_hub as hub
import warnings
# Suppress cosmetic warnings from ART regarding missing optional frameworks (PyTorch)
warnings.filterwarnings("ignore", category=UserWarning, module="art.estimators.certification")

import pkg_resources  # Use the shim in root
from helpers import create_tf_hub_classifier, ensure_labels, setup_logger

# Configure premium colored logging
logger = setup_logger(__name__)

class Classifier:
    def __init__(self):
        self.labels = ensure_labels()
        self.classifier = create_tf_hub_classifier()
        self.model = self.classifier.model

    def classify(self, image_path: Path, forbidden_classes: list) -> Optional[dict]:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        
        img_resized = cv2.resize(image, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        predictions = self.model(img_expanded).numpy()
        top_indices = np.argsort(predictions[0])[-10:][::-1]
        detected_labels = [(self.labels[idx], idx, predictions[0][idx]) for idx in top_indices]

        # Detailed logging per image
        logger.info(f"\n🖼️ Image: {image_path.name}")
        for i, (name, idx, prob) in enumerate(detected_labels):
            logger.info(f"  [{i+1}] {name} (ID {idx}) -> Confidence: {prob:.4f}")

        forbidden_names = {cls.strip().lower() for cls in forbidden_classes}
        matches = []
        for i, (name, idx, prob) in enumerate(detected_labels):
            if name.lower() in forbidden_names:
                matches.append({"name": name, "rank": i + 1, "prob": prob})

        is_forbidden = len(matches) > 0
        if is_forbidden:
            logger.info("❌ Forbidden object detected!")
        else:
            logger.info("✅ Success: No forbidden objects detected.")

        return {
            "image_path": image_path,
            "top_label": detected_labels[0][0] if detected_labels else "Unknown",
            "top_confidence": detected_labels[0][2] if detected_labels else 0.0,
            "is_forbidden": is_forbidden,
            "matches": matches
        }

def process_directory(directory_path: str, forbidden_classes: list):
    dir_path = Path(directory_path)
    if not dir_path.exists():
        logger.error(f"Directory {directory_path} does not exist!")
        return

    image_extensions = {".jpg", ".png", ".jpeg", ".bmp", ".tiff"}
    image_files = [f for f in dir_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"No valid image files found in {directory_path}")
        return

    logger.info(f"Processing {len(image_files)} images in '{directory_path}'...")
    
    classifier = Classifier()
    results = []
    for img_file in image_files:
        result = classifier.classify(img_file, forbidden_classes)
        if result:
            results.append(result)
    
    sorted_results = sorted(results, key=lambda x: x["top_confidence"], reverse=True)
    
    logger.info("\n🔍 Summary (Sorted by Confidence):")
    for res in sorted_results:
        status = "❌" if res["is_forbidden"] else "✅"
        msg = f"{status} {res['image_path'].name} | {res['top_label']} ({res['top_confidence']:.4f})"
        
        if res["is_forbidden"]:
            match_details = ", ".join([f"{m['name']} at #{m['rank']}" for m in res["matches"]])
            msg += f" | FORBIDDEN: {match_details}"
            
        logger.info(msg)

def main():
    parser = argparse.ArgumentParser(description="Classify images and check for forbidden objects.")
    parser.add_argument("path", type=str, help="Path to image file or directory")
    parser.add_argument("--forbidden", type=str, default="keeshond", help="Comma-separated forbidden classes")
    
    args = parser.parse_args()
    forbidden_list = [c.strip() for c in args.forbidden.split(",")]
    
    path = Path(args.path)
    if path.is_dir():
        process_directory(args.path, forbidden_list)
    else:
        classifier = Classifier()
        classifier.classify(path, forbidden_list)

if __name__ == "__main__":
    main()
