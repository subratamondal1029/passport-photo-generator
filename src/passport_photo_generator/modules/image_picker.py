"""Module to simulate FastAPI's UploadFile from local files for testing and CLI usage."""

from __future__ import annotations

import io
import mimetypes
from pathlib import Path
from typing import BinaryIO


class LocalUploadFile:
    """Mock container matching FastAPI/Starlette's UploadFile interface.

    Attributes:
        filename: Name of the uploaded file.
        content_type: MIME type of the file.
        file: Underlying binary stream.
        size: Size of the file in bytes.
    """

    def __init__(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found at: {path.resolve()}")

        self.filename: str = path.name
        guessed_type, _ = mimetypes.guess_type(path)
        self.content_type: str = guessed_type or "application/octet-stream"

        content = path.read_bytes()
        self.size: int = len(content)
        self.file: BinaryIO = io.BytesIO(content)

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the underlying file stream."""
        return self.file.read(size)

    def seek(self, offset: int) -> int:
        """Seek to a byte offset in the underlying stream."""
        return self.file.seek(offset)

    def close(self) -> None:
        """Close the underlying file stream."""
        self.file.close()


def pick_local_image(file_path: str | Path = "data/photo-male.jpg") -> LocalUploadFile:
    """Load a local image into a FastAPI UploadFile-compatible wrapper.

    Args:
        file_path: Path to the local image file. Defaults to 'data/photo.jpg'.

    Returns:
        LocalUploadFile: UploadFile-compatible instance.
    """
    return LocalUploadFile(file_path)
