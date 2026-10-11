"""Passport Photo Generator package and CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from passport_photo_generator.modules import (
    crop_passport_photo,
    detect_face_landmarks,
    export_jpeg,
    pick_local_image,
    process_image_input,
    remove_background,
    replace_background,
    size_passport_photo,
)


def generate_passport_photo(
    image_source: Any,
    output_path: str | Path | None = None,
    width_mm: float = 35.0,
    height_mm: float = 45.0,
    bg_color: tuple[int, int, int] | str = (135, 206, 235),  # Sky blue
    dpi: int = 300,
) -> tuple[np.ndarray, Path | None]:
    """Execute end-to-end pipeline to create a passport photograph.

    Args:
        image_source: LocalUploadFile, FastAPI UploadFile, bytes, or file path.
        output_path: Optional path to save exported JPEG.
        width_mm: Physical width in mm (default 35.0 mm).
        height_mm: Physical height in mm (default 45.0 mm).
        bg_color: Solid background color in RGB or hex (default: Sky Blue).
        dpi: Target resolution in dots per inch (default: 300 DPI).

    Returns:
        tuple[np.ndarray, Path | None]: (Final sized BGR image, saved file path or None).
    """
    # 1. Image input decoding & validation
    bgr_img = process_image_input(image_source)

    # 2. Facial landmark detection
    landmarks = detect_face_landmarks(bgr_img)

    # 3. Background segmentation
    bgra_cutout, _ = remove_background(bgr_img)

    # 4. Horizontal centering & passport-standard cropping
    cropped_cutout = crop_passport_photo(
        bgra_cutout,
        landmarks=landmarks,
        width_mm=width_mm,
        height_mm=height_mm,
    )

    # 5. Background replacement with configurable color (default: Sky Blue)
    composited = replace_background(cropped_cutout, bg_color=bg_color)

    # 6. Sizing to exact physical dimensions at target DPI
    final_photo = size_passport_photo(
        composited,
        width_mm=width_mm,
        height_mm=height_mm,
        dpi=dpi,
    )

    # 7. Export high-resolution JPEG if output path is provided
    saved_path = None
    if output_path is not None:
        saved_path = export_jpeg(final_photo, output_path=output_path, dpi=dpi)

    return final_photo, saved_path


def main() -> None:
    """CLI entrypoint for passport-photo-generator."""
    parser = argparse.ArgumentParser(description="Generate standardized passport-size photos.")
    parser.add_argument(
        "-i", "--input",
        default="data/photo-female.webp",
        help="Input image path (default: data/photo-female.webp)",
    )
    parser.add_argument(
        "-o", "--output",
        default="data/passport_photo.jpg",
        help="Output JPEG path (default: data/passport_photo.jpg)",
    )
    parser.add_argument(
        "-w", "--width",
        type=float,
        default=35.0,
        help="Target physical width in mm (default: 35.0 mm)",
    )
    parser.add_argument(
        "-H", "--height",
        type=float,
        default=45.0,
        help="Target physical height in mm (default: 45.0 mm)",
    )
    parser.add_argument(
        "-c", "--color",
        default="#87CEEB",
        help="Background color in hex or R,G,B (default: #87CEEB Sky Blue)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Print DPI resolution (default: 300)",
    )

    args = parser.parse_args()

    # Parse color if comma-separated
    color_val: tuple[int, int, int] | str = args.color
    if "," in args.color:
        parts = [int(p.strip()) for p in args.color.split(",")]
        if len(parts) == 3:
            color_val = (parts[0], parts[1], parts[2])

    print(f"Loading input image: {args.input}")
    upload_file = pick_local_image(args.input)

    _, saved = generate_passport_photo(
        image_source=upload_file,
        output_path=args.output,
        width_mm=args.width,
        height_mm=args.height,
        bg_color=color_val,
        dpi=args.dpi,
    )
    print(f"Passport photo successfully generated: {saved}")


if __name__ == "__main__":
    main()
