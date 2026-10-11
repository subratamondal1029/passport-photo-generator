# Passport Photo Generator

A Python application and CLI tool that converts a portrait photograph into a standardized, high-resolution passport-size photo ready for printing.

## Workflow

1. **Image Input & Validation:** Loads images (JPEG, PNG, WEBP, AVIF) with automatic EXIF orientation normalization using a container compatible with FastAPI's `UploadFile`.
2. **Face Landmark Detection:** Detects 478 3D facial landmarks using MediaPipe Face Landmarker (`models/face_landmarker.task`) with multi-scale zoom fallback for distant/standing shots.
3. **Background Removal:** Extracts the person and fine hair details using high-accuracy matting via `rembg` (`u2net_human_seg`).
4. **Horizontal Centering & Passport Framing:** Centers the subject horizontally on the nose/face midline and frames according to passport standards (default: Indian 35×45 mm, ~52% head ratio including chest/shoulders, and ~14% top headroom).
5. **Background Replacement:** Composites the subject onto a configurable solid background color (default: Sky Blue `#87CEEB`).
6. **Passport Sizing:** Scales the photo to exact physical dimensions at 300 DPI using Lanczos resampling ($35 \times 45\text{ mm} \rightarrow 413 \times 531\text{ px}$).
7. **High-Resolution Export:** Saves a 300 DPI JPEG with embedded JFIF metadata and 4:4:4 chroma subsampling for print sharpness.

---

## Technology Stack

- **Python:** Core runtime (managed via `uv`).
- **OpenCV:** Image transformations, color conversion, and Lanczos resizing.
- **MediaPipe:** Facial landmark detection and facial geometry.
- **rembg (ONNX Runtime):** Deep learning human foreground segmentation and alpha matting.
- **Pillow:** Image decoding, EXIF orientation handling, and DPI metadata tagging.
- **NumPy:** Multidimensional array processing and alpha blending.

---

## Models

- **`models/face_landmarker.task`**: MediaPipe facial landmark detection model.
- **`u2net_human_seg.onnx`**: High-resolution human matting model (automatically downloaded and cached in `~/.u2net/` on first run).

---

## Project Structure

```text
passport_photo_generator/
├── models/
│   └── face_landmarker.task        # MediaPipe landmark model
├── src/passport_photo_generator/
│   ├── __init__.py                 # End-to-end pipeline & CLI entrypoint
│   └── modules/
│       ├── __init__.py             # Module exports
│       ├── image_picker.py         # FastAPI-compatible local file loader
│       ├── image_input.py          # EXIF correction & BGR array decoding
│       ├── face_detection.py       # Landmark detector with zoom fallback
│       ├── face_crop.py            # Centering & chest-inclusive crop
│       ├── background_removal.py   # High-accuracy matting via rembg
│       ├── background_replacement.py # Solid background replacement
│       ├── passport_sizing.py      # Physical dimensions & 300 DPI scaler
│       └── export.py               # High-res JPEG exporter with DPI metadata
├── data/                           # Input and output photos
├── pyproject.toml
└── uv.lock
```

---

## Quick Start & CLI Usage

### Installation

Ensure `uv` is installed, then sync project dependencies:

```bash
uv sync
```

### Run CLI

Convert an image with default settings (Indian standard 35×45 mm, sky blue background, 300 DPI):

```bash
uv run passport-photo-generator -i data/my-photo.jpg -o data/my-photo-output.jpg
```

### Custom Options

```bash
# White background (#FFFFFF), US passport 2x2 inch (50.8 x 50.8 mm) at 300 DPI
uv run passport-photo-generator \
  -i data/photo.jpg \
  -o data/output.jpg \
  -w 50.8 \
  -H 50.8 \
  -c "#FFFFFF" \
  --dpi 300

# Custom RGB background (e.g., light blue: 173,216,230)
uv run passport-photo-generator \
  -i data/photo.jpg \
  -o data/output.jpg \
  -c "173,216,230"
```

### CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--input` | `data/photo-female.webp` | Path to the source photo. |
| `-o`, `--output` | `data/passport_photo.jpg` | Path for the exported JPEG file. |
| `-w`, `--width` | `35.0` | Target physical width in mm. |
| `-H`, `--height` | `45.0` | Target physical height in mm. |
| `-c`, `--color` | `#87CEEB` | Background color (hex string or `R,G,B`). |
| `--dpi` | `300` | Output print resolution in DPI. |