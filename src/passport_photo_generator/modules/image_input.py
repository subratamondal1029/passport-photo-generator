"""Module for decoding and validating uploaded image files into OpenCV arrays."""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


def process_image_input(upload_file: Any) -> np.ndarray:
    """Read an UploadFile or binary stream and decode into an OpenCV BGR ndarray.

    Handles EXIF orientation tags automatically and validates minimum dimensions.

    Args:
        upload_file: FastAPI UploadFile, file-like object with read(), or raw bytes.

    Returns:
        np.ndarray: Decoded image in standard BGR color format (H, W, 3).

    Raises:
        ValueError: If file content cannot be decoded or image is too small.
    """
    if hasattr(upload_file, "read"):
        raw_bytes = upload_file.read()
        if hasattr(upload_file, "seek"):
            upload_file.seek(0)
    elif isinstance(upload_file, (bytes, bytearray)):
        raw_bytes = bytes(upload_file)
    else:
        raise ValueError("Unsupported input format: expected UploadFile, file-like object, or bytes.")

    if not raw_bytes:
        raise ValueError("Provided image input is empty.")

    try:
        pil_image = Image.open(io.BytesIO(raw_bytes))
        # Correct image rotation based on EXIF orientation metadata
        pil_image = ImageOps.exif_transpose(pil_image)
        # Convert to RGB if RGBA, grayscale, or palette
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
    except Exception as err:
        raise ValueError(f"Failed to decode image data: {err}") from err

    # Convert PIL (RGB) to OpenCV (BGR) array
    rgb_array = np.array(pil_image)
    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    height, width = bgr_image.shape[:2]
    if height < 100 or width < 100:
        raise ValueError(f"Image resolution too low ({width}x{height}). Minimum required is 100x100.")

    return bgr_image
