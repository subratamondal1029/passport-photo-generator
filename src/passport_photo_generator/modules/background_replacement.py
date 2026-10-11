"""Module for replacing image background with solid configurable colors."""

from __future__ import annotations

import cv2
import numpy as np

# Default: Sky Blue in RGB
DEFAULT_BG_COLOR = (135, 206, 235)


def _parse_color(color: tuple[int, int, int] | str) -> tuple[int, int, int]:
    """Parse color into an (R, G, B) tuple from hex string or 3-tuple."""
    if isinstance(color, str):
        hex_str = color.lstrip("#")
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return (r, g, b)
        raise ValueError(f"Invalid hex color string: {color}")
    if isinstance(color, (tuple, list)) and len(color) == 3:
        return (int(color[0]), int(color[1]), int(color[2]))
    raise ValueError(f"Unsupported color format: {color}. Expected RGB tuple or hex string.")


def replace_background(
    bgra_image: np.ndarray,
    bg_color: tuple[int, int, int] | str = DEFAULT_BG_COLOR,
) -> np.ndarray:
    """Composite subject cutout onto a solid background color.

    Args:
        bgra_image: Subject image with alpha channel (H, W, 4).
        bg_color: Background color as (R, G, B) tuple or hex code. Defaults to Sky Blue (135, 206, 235).

    Returns:
        np.ndarray: Composited 3-channel BGR image (H, W, 3), uint8.
    """
    if bgra_image.ndim != 3 or bgra_image.shape[2] != 4:
        raise ValueError(f"Expected BGRA 4-channel image, got shape: {bgra_image.shape}")

    rgb_color = _parse_color(bg_color)
    bgr_bg = np.array([rgb_color[2], rgb_color[1], rgb_color[0]], dtype=np.float32)

    h, w = bgra_image.shape[:2]
    foreground_bgr = bgra_image[:, :, :3].astype(np.float32)
    alpha = (bgra_image[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    # Solid background canvas in BGR
    background = np.full((h, w, 3), bgr_bg, dtype=np.float32)

    # Alpha blending: output = fg * alpha + bg * (1 - alpha)
    blended = (foreground_bgr * alpha) + (background * (1.0 - alpha))
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
