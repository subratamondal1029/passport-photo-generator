"""Background segmentation and removal module using MediaPipe SelfieSegmenter."""

from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def _create_segmenter(model_path: str | Path) -> vision.ImageSegmenter:
    """Create a MediaPipe ImageSegmenter instance using selfie_segmenter.tflite."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Segmenter model not found at: {path.resolve()}")

    base_options = python.BaseOptions(model_asset_path=str(path))
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


def remove_background(
    bgr_image: np.ndarray,
    model_path: str | Path = "models/selfie_segmenter.tflite",
    threshold: float = 0.5,
    feather_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment person from background and generate transparent cutout and alpha mask.

    Args:
        bgr_image: Input image as OpenCV BGR ndarray (H, W, 3).
        model_path: Path to selfie_segmenter.tflite model file.
        threshold: Confidence threshold for subject segmentation (0.0 to 1.0).
        feather_edges: Whether to gently blur mask borders to avoid jagged edges.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - bgra_image: Cutout with alpha channel (H, W, 4), uint8.
            - alpha_mask: Continuous alpha mask (H, W), float32 in range [0.0, 1.0].
    """
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    with _create_segmenter(model_path) as segmenter:
        result = segmenter.segment(mp_img)

    raw_mask = result.confidence_masks[0].numpy_view()
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[:, :, 0]

    # Normalize mask values based on threshold
    # Soft sigmoid-style edge ramp around threshold
    if feather_edges:
        # Slight Gaussian blur to soften harsh pixelation around hair
        smoothed = cv2.GaussianBlur(raw_mask, (5, 5), sigmaX=1.0)
        alpha_mask = np.clip((smoothed - (threshold - 0.15)) / 0.3, 0.0, 1.0).astype(np.float32)
    else:
        alpha_mask = (raw_mask >= threshold).astype(np.float32)

    # Build BGRA cutout
    alpha_uint8 = (alpha_mask * 255.0).astype(np.uint8)
    bgra_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
    bgra_image[:, :, 3] = alpha_uint8

    return bgra_image, alpha_mask
