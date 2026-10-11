"""Module for scaling photos to precise physical dimensions and DPI specifications."""

from __future__ import annotations

import cv2
import numpy as np

MM_PER_INCH = 25.4
DEFAULT_WIDTH_MM = 35.0
DEFAULT_HEIGHT_MM = 45.0
DEFAULT_DPI = 300


def mm_to_pixels(mm_size: float, dpi: int = DEFAULT_DPI) -> int:
    """Convert millimeter physical size to pixel dimensions at given DPI."""
    return int(round((mm_size / MM_PER_INCH) * dpi))


def size_passport_photo(
    image: np.ndarray,
    width_mm: float = DEFAULT_WIDTH_MM,
    height_mm: float = DEFAULT_HEIGHT_MM,
    dpi: int = DEFAULT_DPI,
) -> np.ndarray:
    """Resize image to target physical dimensions at specified DPI.

    Args:
        image: Input image (H, W, C) in BGR or BGRA format.
        width_mm: Physical width in mm (default: 35.0 mm).
        height_mm: Physical height in mm (default: 45.0 mm).
        dpi: Target dots per inch resolution (default: 300 DPI).

    Returns:
        np.ndarray: Rescaled image matching exact pixel dimensions for target standard.
    """
    target_width = mm_to_pixels(width_mm, dpi)
    target_height = mm_to_pixels(height_mm, dpi)

    current_h, current_w = image.shape[:2]
    # Use Lanczos for upscaling, Area for downscaling
    interpolation = cv2.INTER_AREA if (target_width < current_w and target_height < current_h) else cv2.INTER_LANCZOS4

    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return resized
