"""Module for exporting high-resolution JPEG images with DPI metadata."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def export_jpeg(
    bgr_image: np.ndarray,
    output_path: str | Path,
    dpi: int = 300,
    quality: int = 95,
) -> Path:
    """Save OpenCV BGR image as high-quality JPEG with embedded DPI metadata.

    Args:
        bgr_image: Image to export as OpenCV BGR ndarray.
        output_path: File path where the JPEG will be saved.
        dpi: Target DPI metadata to embed in JFIF header (default: 300).
        quality: JPEG compression quality from 1 to 100 (default: 95).

    Returns:
        Path: Resolved path of the saved JPEG file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert BGR OpenCV image to RGB PIL Image
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # Save with embedded DPI metadata and maximum quality
    pil_image.save(
        path,
        format="JPEG",
        dpi=(dpi, dpi),
        quality=quality,
        subsampling=0,  # 4:4:4 chroma subsampling for crisp edge retention
    )
    return path
