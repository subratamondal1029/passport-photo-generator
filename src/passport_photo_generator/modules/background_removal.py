"""Background segmentation and removal module using rembg."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np
from rembg import new_session, remove

DEFAULT_MODEL_NAME = "u2net_human_seg"
_GLOBAL_SESSION: Any = None


def get_rembg_session(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Retrieve or initialize a cached rembg ONNX session."""
    global _GLOBAL_SESSION
    if _GLOBAL_SESSION is None or getattr(_GLOBAL_SESSION, "model_name", None) != model_name:
        _GLOBAL_SESSION = new_session(model_name)
    return _GLOBAL_SESSION


def remove_background(
    bgr_image: np.ndarray,
    model_name: str = DEFAULT_MODEL_NAME,
    alpha_matting: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment person from background using rembg and generate high-fidelity cutout.

    Args:
        bgr_image: Input image as OpenCV BGR ndarray (H, W, 3).
        model_name: Rembg model name. Defaults to 'u2net_human_seg'.
        alpha_matting: Enable alpha matting for enhanced hair and edge precision.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - bgra_image: Subject cutout with alpha channel (H, W, 4), uint8.
            - alpha_mask: Continuous alpha mask (H, W), float32 in range [0.0, 1.0].

    Raises:
        RuntimeError: If background removal fails.
    """
    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError(f"Expected BGR 3-channel image, got shape: {bgr_image.shape}")

    try:
        session = get_rembg_session(model_name)

        # rembg expects RGB input
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Run rembg
        rgba_output = remove(
            rgb_image,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )

        if not isinstance(rgba_output, np.ndarray):
            rgba_output = np.array(rgba_output)

        # Convert RGBA to BGRA for OpenCV consistency
        bgra_image = cv2.cvtColor(rgba_output, cv2.COLOR_RGBA2BGRA)

        # Extract normalized float32 alpha mask [0.0, 1.0]
        alpha_channel = bgra_image[:, :, 3].astype(np.float32) / 255.0

        return bgra_image, alpha_channel

    except Exception as err:
        raise RuntimeError(f"Background removal failed using model '{model_name}': {err}") from err
