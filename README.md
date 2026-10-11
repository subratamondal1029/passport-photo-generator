# Passport Photo Sheet Generator

## Goal

Build a Python-based application that takes a single person's photo, processes it into a standardized passport-size photograph
## Workflow

1. **Image Input:** Accept a single image from the user.
2. **Face Detection:** Use MediaPipe Face Landmarker to detect facial landmarks.
3. **Face Alignment:** Position the face centrally and correct its orientation.
4. **Cropping and Resizing:** Crop the image to maintain consistent face proportions and framing.
5. **Background Removal:** Use the MediaPipe SelfieSegmenter model to separate the person from the original background.
6. **Background Replacement:** Replace the original background with a configurable solid color, such as white or blue.
7. **Passport Photo Formatting:** Apply the selected passport photo standard, including physical dimensions and head-position requirements.
8. **Export:** Output a high resolution JPEG image 

## Technology Stack

- **Python:** Main programming language.
- **OpenCV:** Image processing, cropping, rotation, and resizing.
- **MediaPipe Face Landmarker:** Facial landmark detection and alignment.
- **MediaPipe SelfieSegmenter:** Person segmentation and background removal.
- **NumPy:** Image-array and segmentation-mask processing.
- **Pillow:** Background compositing, precise photo dimensions

## Models

- `face_landmarker.task` — facial landmark detection.
- `selfie_segmenter.tflite` — 256 × 256, float16, square — background segmentation.

## Initial Scope

The first version will support one input photo, one selected passport photo standard, a configurable background color.


## Important Constraints

- A single photograph cannot reliably reveal a person's true physical dimensions or camera distance. The application will standardize visible face size and composition using facial landmarks and configured framing rules.
- Passport photo dimensions and facial-position requirements depend on a constant data.
- Background segmentation quality, especially around hair, must be tested before using the output for printing.