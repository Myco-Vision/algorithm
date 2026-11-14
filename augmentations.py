"""
augmentations.py — Mushroom Field Photography Data Augmentation
================================================================
Augmentations tailored for outdoor mushroom photography:
  1. Geometric (configured in ImageDataGenerator):
     - Random rotation: ±40°
     - Width & height shifts: ±20%
     - Zoom: ±20%
     - Shear: 0.2
     - Horizontal flip: enabled
     - Vertical flip: DISABLED (preserves natural upward stipe/cap orientation)
     - Border fill mode: 'reflect' (eliminates edge-smearing / stretched pixel artifacts)
  2. Photometric & Exposure (in field_photometry_and_occlusion):
     - Brightness & Contrast Jitter: ±15% to ±25% for dappled forest sunlight & shade
     - Subtle Saturation (±15%) & minimal Hue shift (±0.03 max) to simulate
       phone camera white-balance without altering species-defining cap/gill colors
  3. Field Artifacts & Occlusion (in field_photometry_and_occlusion):
     - Subtle Gaussian Blur: ~25% probability, sigma 0.3-1.0 (autofocus miss)
     - Random Erasing / Cutout: p=0.5, 1 patch, 2%-8% area with random noise fill
       (simulates leaves, twigs, moss, or grass without obscuring species features)

Applied strictly to the training pipeline; validation/testing pipelines remain
clean with standard resizing and normalization only.
"""

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter


def field_photometry_and_occlusion(img: np.ndarray) -> np.ndarray:
    """
    Applies outdoor field augmentations to a single image (H, W, 3) in [0, 255].
    Passed as preprocessing_function to Keras ImageDataGenerator.
    """
    img = img.copy()
    h, w, c = img.shape
    total_area = h * w

    # 1. Contrast Jitter (±15% to ±20%)
    contrast_factor = np.random.uniform(0.80, 1.20)
    mean_val = np.mean(img, axis=(0, 1), keepdims=True)
    img = np.clip((img - mean_val) * contrast_factor + mean_val, 0.0, 255.0)

    # 2. Subtle Saturation (±15%) & Minimal Hue Shift (±0.02 to ±0.03 max)
    # Preserves species-identifying cap, gill, and spore pigmentation
    pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    sat_factor = np.random.uniform(0.85, 1.15)
    pil_img = ImageEnhance.Color(pil_img).enhance(sat_factor)

    # Hue adjustment in HSV color space (delta clamped to [-0.03, +0.03])
    hsv = np.array(pil_img.convert("HSV"), dtype=np.float32)
    hue_delta = np.random.uniform(-0.03, 0.03) * 255.0
    hsv[..., 0] = (hsv[..., 0] + hue_delta) % 256.0
    img = np.array(Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB"), dtype=np.float32)

    # 3. Subtle Gaussian Blur (~25% probability, sigma 0.3 to 1.0)
    # Simulates slight smartphone macro / autofocus misses on field close-ups
    if np.random.rand() < 0.25:
        sigma = np.random.uniform(0.3, 1.0)
        img = gaussian_filter(img, sigma=(sigma, sigma, 0))

    # 4. Random Erasing / Cutout (p=0.5, 1 patch, 2%-8% image area, random noise fill)
    # Simulates leaves, twigs, moss, or grass partially occluding the mushroom without obscuring key features
    if np.random.rand() < 0.5:
        patch_ratio = np.random.uniform(0.02, 0.08)
        patch_area = total_area * patch_ratio
        aspect_ratio = np.random.uniform(0.5, 2.0)
        patch_h = int(np.sqrt(patch_area * aspect_ratio))
        patch_w = int(np.sqrt(patch_area / aspect_ratio))
        patch_h = min(max(patch_h, 4), h - 1)
        patch_w = min(max(patch_w, 4), w - 1)

        y = np.random.randint(0, h - patch_h)
        x = np.random.randint(0, w - patch_w)
        noise = np.random.uniform(0.0, 255.0, (patch_h, patch_w, c)).astype(np.float32)
        img[y:y + patch_h, x:x + patch_w, :] = noise

    return img
