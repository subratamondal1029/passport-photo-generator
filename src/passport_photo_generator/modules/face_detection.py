"""Face and facial landmark detection module using MediaPipe FaceLandmarker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass(frozen=True)
class FaceLandmarks:
    """Container for detected facial landmarks in absolute pixel coordinates.

    Coordinates (x, y) are in the coordinate frame of the original input image.
    """

    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    chin: tuple[float, float]
    forehead_top: tuple[float, float]
    left_ear: tuple[float, float]
    right_ear: tuple[float, float]
    bounding_box: tuple[int, int, int, int]  # (x, y, w, h)
    all_landmarks: np.ndarray  # shape: (N, 3), [x, y, z] in pixel units


# Key landmark indices in the MediaPipe 468/478 Face Mesh topology
LANDMARK_NOSE_TIP = 1
LANDMARK_FOREHEAD_TOP = 10
LANDMARK_CHIN = 152
LANDMARK_LEFT_EYE_OUTER = 33
LANDMARK_RIGHT_EYE_OUTER = 263
LANDMARK_LEFT_EAR = 234
LANDMARK_RIGHT_EAR = 454


def _create_detector(model_path: str | Path, min_confidence: float = 0.3) -> vision.FaceLandmarker:
    """Instantiate a MediaPipe FaceLandmarker object."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at: {path.resolve()}")

    base_options = python.BaseOptions(model_asset_path=str(path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
        num_faces=2,  # detect up to 2 to catch multiple-face violations
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_face_landmarks(
    bgr_image: np.ndarray,
    model_path: str | Path = "models/face_landmarker.task",
    min_confidence: float = 0.3,
) -> FaceLandmarks:
    """Detect exactly one face and extract facial landmarks in pixel coordinates.

    Supports fallback upscaling for full-body / distant shots where the face is small.

    Args:
        bgr_image: Input image as an OpenCV BGR numpy array.
        model_path: Path to the MediaPipe face_landmarker.task model.
        min_confidence: Minimum detection confidence threshold.

    Returns:
        FaceLandmarks: Structured landmark points and bounding box.

    Raises:
        ValueError: If 0 faces or more than 1 face are detected.
    """
    orig_h, orig_w = bgr_image.shape[:2]

    with _create_detector(model_path, min_confidence=min_confidence) as detector:
        # Pass 1: Direct detection on full frame
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = detector.detect(mp_img)

        # Pass 2 Fallback: If 0 faces detected and image has vertical aspect ratio,
        # try the upper half (head region in standing shots) with 2x zoom.
        offset_x, offset_y, scale_factor = 0.0, 0.0, 1.0
        if len(result.face_landmarks) == 0:
            upper_h = int(orig_h * 0.45)
            upper_crop = bgr_image[0:upper_h, :]
            scale_factor = 2.0
            zoomed = cv2.resize(upper_crop, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
            mp_zoomed = mp.Image(image_format=mp.ImageFormat.SRGB, data=zoomed_rgb)
            result = detector.detect(mp_zoomed)
            # Coordinates in zoomed crop will map back via / scale_factor

        detected_count = len(result.face_landmarks)
        if detected_count == 0:
            raise ValueError("No face detected in the image. Please provide a clear portrait photo.")
        if detected_count > 1:
            raise ValueError(f"Multiple faces detected ({detected_count}). Passport photos require exactly one person.")

        raw_landmarks = result.face_landmarks[0]

        # Convert normalized coordinates to input image pixel space
        if scale_factor != 1.0:
            # Map back from zoomed upper crop
            crop_h = int(orig_h * 0.45) * scale_factor
            crop_w = orig_w * scale_factor
            coords = np.array(
                [[(lm.x * crop_w) / scale_factor, (lm.y * crop_h) / scale_factor, lm.z] for lm in raw_landmarks],
                dtype=np.float32,
            )
        else:
            coords = np.array(
                [[lm.x * orig_w, lm.y * orig_h, lm.z] for lm in raw_landmarks],
                dtype=np.float32,
            )

        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        x_min, x_max = int(np.clip(np.min(x_coords), 0, orig_w)), int(np.clip(np.max(x_coords), 0, orig_w))
        y_min, y_max = int(np.clip(np.min(y_coords), 0, orig_h)), int(np.clip(np.max(y_coords), 0, orig_h))
        bbox = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))

        return FaceLandmarks(
            left_eye=(float(coords[LANDMARK_LEFT_EYE_OUTER, 0]), float(coords[LANDMARK_LEFT_EYE_OUTER, 1])),
            right_eye=(float(coords[LANDMARK_RIGHT_EYE_OUTER, 0]), float(coords[LANDMARK_RIGHT_EYE_OUTER, 1])),
            nose_tip=(float(coords[LANDMARK_NOSE_TIP, 0]), float(coords[LANDMARK_NOSE_TIP, 1])),
            chin=(float(coords[LANDMARK_CHIN, 0]), float(coords[LANDMARK_CHIN, 1])),
            forehead_top=(float(coords[LANDMARK_FOREHEAD_TOP, 0]), float(coords[LANDMARK_FOREHEAD_TOP, 1])),
            left_ear=(float(coords[LANDMARK_LEFT_EAR, 0]), float(coords[LANDMARK_LEFT_EAR, 1])),
            right_ear=(float(coords[LANDMARK_RIGHT_EAR, 0]), float(coords[LANDMARK_RIGHT_EAR, 1])),
            bounding_box=bbox,
            all_landmarks=coords,
        )
