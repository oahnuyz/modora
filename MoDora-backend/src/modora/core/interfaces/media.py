from typing import Protocol

from modora.core.domain.component import Location


class ImageProvider(Protocol):
    """Image provider interface protocol.

    Defines the standard interface for obtaining image data from source files (e.g., PDFs).
    """

    def crop_image(self, source: str, locations: list[Location]) -> list[str]:
        """Crop images from the source file based on location information.

        Args:
            source: The source file path or identifier.
            locations: The location list to crop.

        Returns:
            list[str]: Base64 encoded image strings.
        """
        ...
