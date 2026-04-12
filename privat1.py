#!/usr/bin/env python3
"""
Privat1 - Advanced Adversarial Image Protection (2026 Edition)
Modernized for robust AI obfuscation, performance, and ease of use.
"""

import os
# Suppress TensorFlow C++ internal logs before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import logging
import random
import string
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

# Suppress cosmetic warnings from ART regarding missing optional frameworks (PyTorch)
warnings.filterwarnings("ignore", category=UserWarning, module="art.estimators.certification")

from helpers import create_tf_hub_classifier, setup_logger, box_msg

# Modular imports from src/
from src.attacks.PGD import apply_pgd_with_upsized_delta
from src.attacks.FGM import apply_fgm_with_upsized_delta
from src.processing.PSPM import (
    apply_pixel_shift, 
    apply_pixel_pattern_mask, 
    apply_random_perspective_transform,
    embed_distractor_assets
)
from src.processing.NBP import pixelate as apply_pixelation
from src.processing.steg import add_confusing_metadata

# Configure premium colored logging
logger = setup_logger(__name__)

############################################
#               HELPER FUNCS               #
############################################

def load_image(image_path: Path) -> np.ndarray:
    """Loads image via OpenCV."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    return image

def generate_random_name(extension: str) -> str:
    """Generates a random 16-hex-digit filename."""
    random_name = ''.join(random.choices(string.hexdigits.lower(), k=16))
    return f"{random_name}.{extension}"

def remove_metadata(image_source, output_path: Path):
    """
    Strips ALL metadata by re-saving raw pixel data with default PIL settings.
    Accepts either a path or a numpy array.
    """
    if isinstance(image_source, np.ndarray):
        # Convert BGR to RGB for PIL
        image = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
    else:
        image = Image.open(image_source)
    
    # Rebuilding the image from raw data strips all markers (EXIF, XMP, IPTC, ICC)
    data = list(image.getdata())
    image_no_metadata = Image.new(image.mode, image.size)
    image_no_metadata.putdata(data)
    image_no_metadata.save(output_path, "JPEG", quality=95, subsampling=0)
    logger.info(f"Metadata-stripped image saved to {output_path}")

############################################
#       ADVERSARIAL ATTACK DISPATCHER      #
############################################

def apply_adversarial_step(
    image: np.ndarray, 
    attack_type: str = "pgd", 
    eps: float = 0.04, 
    iters: int = 15,
    confidence: float = 1.0,
    target_label: int = 865
) -> np.ndarray:
    """
    Dispatcher for modular adversarial attacks.
    """
    if attack_type == "pgd":
        return apply_pgd_with_upsized_delta(image, epsilon=eps, max_iter=iters, target_label=target_label)
    elif attack_type == "fgm":
        return apply_fgm_with_upsized_delta(image, epsilon=eps, target_label=target_label)
    else:
        logger.warning(f"Unknown attack type {attack_type}, skipping.")
        return image

############################################
#            MAIN PROTECTION FLOW          #
############################################

def protect_image(input_path: Path, output_dir: Path, args: argparse.Namespace, batch_info: tuple = (0, 0)) -> Path:
    """
    Complete protection pipeline: 
    1. Distortions -> 2. Adversarial -> 3. Metadata Stripping
    """
    idx, total = batch_info
    print(box_msg(f"[{idx}/{total}] PROCESSING: {input_path.name}"))
    
    image = load_image(input_path)

    # 1. Structural Distortions
    if not args.no_distort:
        image = apply_pixelation(image, pixel_size=args.pixel_size)
        image = apply_pixel_shift(image, shift_amount=args.shift)
        image = apply_pixel_pattern_mask(image, opacity=args.mask_opacity)
        image = apply_random_perspective_transform(image)
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # 2. Adversarial Attacks
    if not args.no_adv:
        # Chaining PGD and FGM for maximum disruption
        image = apply_adversarial_step(image, attack_type="pgd", eps=args.eps, iters=8)
        image = apply_adversarial_step(image, attack_type="fgm", eps=args.eps * 0.8)

    # 3. Assets/Distractors
    if args.assets:
        image = embed_distractor_assets(image, Path(args.assets))

    # 4. Save and Strip Metadata
    output_name = generate_random_name("jpg")
    final_output = output_dir / output_name
    
    # We save a version with stego first if requested, but final output is always stripped
    remove_metadata(image, final_output)
    
    return final_output

def process_batch(input_dir: Path, output_dir: Path, args: argparse.Namespace):
    """Processes all images in a directory using a thread pool."""
    extensions = {".jpg", ".png", ".jpeg", ".bmp", ".tiff"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return

    total = len(image_files)
    logger.info(f"Batch processing {total} images with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(protect_image, img, output_dir, args, (i+1, total)): img 
                   for i, img in enumerate(image_files)}
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing {futures[future].name}: {e}")

############################################
#               ENTRY POINT                #
############################################

def main():
    parser = argparse.ArgumentParser(description="Privat1 Advanced Image Protection")
    parser.add_argument("--input", type=str, default="images/", help="Input directory")
    parser.add_argument("--output", type=str, default="converted/protected/", help="Output directory")
    parser.add_argument("--eps", type=float, default=0.05, help="Adversarial epsilon strength")
    parser.add_argument("--shift", type=int, default=4, help="Pixel shift amount")
    parser.add_argument("--pixel-size", type=int, default=3, help="Pixelation size")
    parser.add_argument("--mask-opacity", type=float, default=0.2, help="Pattern mask opacity")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument("--assets", type=str, help="Optional path to training/distractor images")
    parser.add_argument("--no-adv", action="store_true", help="Disable adversarial attacks")
    parser.add_argument("--no-distort", action="store_true", help="Disable structural distortions")
    
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} not found.")
        sys.exit(1)

    process_batch(input_dir, output_dir, args)
    logger.info("All images processed successfully.")

if __name__ == "__main__":
    main()
