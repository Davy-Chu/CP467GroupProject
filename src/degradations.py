"""Shared RGB/grayscale transforms; never write transformed datasets to disk.

Accept uint8 [0,255] or floating point [0,1] images (H,W) or RGB (H,W,3).
Return a new array with the same shape/dtype/range; input is never mutated.
Run degradations on the shared resized RGB image BEFORE model-specific steps.
"""

import hashlib

import cv2
import numpy as np

from .config import CONDITIONS, DEGRADATION_PARAMS, RANDOM_SEED


def _normalized(image):
    image = np.asarray(image)
    if image.size == 0 or not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError("Expected a nonempty grayscale (H,W) or RGB (H,W,3) image.")
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError("Images must be uint8 [0,255] or floating point [0,1].")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Floating point images must be finite and in [0,1].")
    return image.astype(np.float64, copy=True)


def _restore(values, original):
    values = np.clip(values, 0, 1)
    if original.dtype == np.uint8:
        return np.rint(values * 255).astype(np.uint8)
    return values.astype(original.dtype)


def reduce_brightness(image, factor=DEGRADATION_PARAMS["brightness"]["factor"]):
    """Multiply intensity by factor in [0,1]; 0.5 halves the brightness."""
    if not np.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError("factor must be finite and between 0 and 1.")
    image = np.asarray(image)
    return _restore(_normalized(image) * factor, image)


def add_noise(image, std=DEGRADATION_PARAMS["noise"]["std"], seed=None):
    """Add zero-mean Gaussian noise; std is on [0,1] (0.05 ~ 12.75/255).

    Supply a seed to reproduce noise without changing the global NumPy RNG.
    For evaluation, use apply_condition to derive a stable seed per image.
    """
    if not np.isfinite(std) or std < 0:
        raise ValueError("std must be finite and nonnegative.")
    image = np.asarray(image)
    values = _normalized(image)
    noise = np.random.default_rng(seed).normal(0, std, size=values.shape)
    return _restore(values + noise, image)


def _kernel_size(size):
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1 or size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")


def apply_motion_blur(image, kernel_size=DEGRADATION_PARAMS["motion_blur"]["kernel_size"]):
    """Horizontal motion blur with an odd length in pixels (default 9)."""
    _kernel_size(kernel_size)
    image = np.asarray(image)
    values = _normalized(image)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return _restore(cv2.filter2D(values, -1, kernel, borderType=cv2.BORDER_REFLECT_101), image)


def equalize_histogram(image):
    """Optional 8-bit histogram equalization; RGB uses YCrCb luminance only.

    Floating inputs are quantized to 8-bit for equalization then restored to
    their original dtype in [0,1]. This is never applied automatically.
    """
    image = np.asarray(image)
    values = np.rint(_normalized(image) * 255).astype(np.uint8)
    if values.ndim == 2:
        result = cv2.equalizeHist(values)
    else:
        ycrcb = cv2.cvtColor(values, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return _restore(result.astype(np.float64) / 255, image)


def gaussian_filter(image, kernel_size=3, sigma=1.0):
    """Optional Gaussian smoothing; odd window and sigma in pixels."""
    _kernel_size(kernel_size)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")
    image = np.asarray(image)
    values = _normalized(image)
    result = cv2.GaussianBlur(values, (kernel_size, kernel_size), sigma, borderType=cv2.BORDER_REFLECT_101)
    return _restore(result, image)


def apply_condition(image, condition, image_id, seed=RANDOM_SEED):
    """Apply the team's default condition with order-independent noise.

    image_id must be the CSV-relative path (e.g. Test/00000.png), identical
    across notebooks. Use it even after shuffling/batching; never use a batch
    index or an absolute Drive/local path as the ID.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}.")
    if condition == "clean":
        _normalized(image)
        return np.asarray(image).copy()
    if condition == "brightness":
        return reduce_brightness(image, **DEGRADATION_PARAMS[condition])
    if condition == "motion_blur":
        return apply_motion_blur(image, **DEGRADATION_PARAMS[condition])
    identity = str(image_id).replace("\\", "/")
    token = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    image_seed = int.from_bytes(token[:8], "little")
    return add_noise(image, seed=image_seed, **DEGRADATION_PARAMS[condition])
