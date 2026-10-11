"""Module for horizontally centering and passport-standard cropping."""

from __future__ import annotations

import cv2
import numpy as np

from passport_photo_generator.modules.face_detection import FaceLandmarks

# Indian passport default standard: 35mm x 45mm
DEFAULT_WIDTH_MM = 35.0
DEFAULT_HEIGHT_MM = 45.0
DEFAULT_HEAD_RATIO = 0.52  # Head occupies ~50-55% of vertical frame (includes chest/shoulders)
DEFAULT_TOP_MARGIN_RATIO = 0.14  # ~14% margin above crown


def crop_passport_photo(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    width_mm: float = DEFAULT_WIDTH_MM,
    height_mm: float = DEFAULT_HEIGHT_MM,
    head_ratio: float = DEFAULT_HEAD_RATIO,
    top_margin_ratio: float = DEFAULT_TOP_MARGIN_RATIO,
) -> np.ndarray:
    """Center the person horizontally and crop according to passport specifications.

    Args:
        image: Input image (BGR or BGRA format).
        landmarks: Detected facial landmarks from face_detection module.
        width_mm: Target physical width in mm. Defaults to 35.0 mm (Indian Standard).
        height_mm: Target physical height in mm. Defaults to 45.0 mm (Indian Standard).
        head_ratio: Proportion of photo height occupied by head (crown to chin).
        top_margin_ratio: Margin above the crown relative to photo height.

    Returns:
        np.ndarray: Centered and cropped image in the target aspect ratio.
    """
    orig_h, orig_w = image.shape[:2]
    num_channels = image.shape[2] if image.ndim == 3 else 1

    # 1. Estimate head height (chin to crown)
    chin_y = landmarks.chin[1]
    forehead_y = landmarks.forehead_top[1]
    forehead_to_chin = max(10.0, chin_y - forehead_y)

    # Approximate top of skull/hair (~25% above forehead)
    head_height = forehead_to_chin * 1.25
    crown_y = chin_y - head_height

    # 2. Compute required crop box dimensions
    crop_h = head_height / max(0.1, head_ratio)
    aspect_ratio = width_mm / height_mm
    crop_w = crop_h * aspect_ratio

    # 3. Compute crop bounds (horizontal center aligned to nose tip)
    center_x = landmarks.nose_tip[0]
    top_y = crown_y - (crop_h * top_margin_ratio)
    left_x = center_x - (crop_w / 2.0)

    crop_x1 = int(round(left_x))
    crop_y1 = int(round(top_y))
    crop_x2 = int(round(left_x + crop_w))
    crop_y2 = int(round(top_y + crop_h))

    actual_w = crop_x2 - crop_x1
    actual_h = crop_y2 - crop_y1

    # 4. Canvas creation to handle out-of-boundary padding transparently
    canvas = np.zeros((actual_h, actual_w, num_channels), dtype=image.dtype)

    # Source overlap coordinates
    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(orig_w, crop_x2)
    src_y2 = min(orig_h, crop_y2)

    # Destination overlap coordinates on canvas
    dst_x1 = src_x1 - crop_x1
    dst_y1 = src_y1 - crop_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    return canvas
