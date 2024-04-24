from dataclasses import dataclass, field
from PIL import Image as Img
from typing import Any, Dict, List

import numpy as np

from pdf2table.table.structure import TableObject
from .image import Image

@dataclass
class Page:
    image_data: Any
    page_number: int
    metadata: Dict[str, Any]
    image: Image = field(init=False)

    def __post_init__(self):
        """Initialize the Image class internally depending on the input type."""
        if isinstance(self.image_data, Image):
            self.image = self.image_data
        elif isinstance(self.image_data, Img.Image):
            self.image = Image(image_array=np.array(self.image_data))
        elif isinstance(self.image_data, str):
            self.image = Image(path=self.image_data)
        else:
            raise TypeError("Unsupported type for image_data. Must be Image, PIL.Image, or file path.")

    def display_image(self):
        """Display the image using the default image viewer."""
        self.image.display_image()

    def get_metadata(self, key: str) -> Any:
        """Retrieve a piece of metadata using a key."""
        return self.metadata.get(key, None)

    def set_metadata(self, key: str, value: Any):
        """Set a piece of metadata."""
        self.metadata[key] = value

    def save_image(self, path):
        """Save the page image to a specified path."""
        self.image.save_image(path)

    def extract_tables(self, implicit_rows: bool = False, borderless_tables: bool = False,
                       min_confidence: int = 50) -> List[TableObject]:
        """Extract tables from the page."""
        return self.image.extract_tables(implicit_rows=implicit_rows, borderless_tables=borderless_tables,
                                         min_confidence=min_confidence)

    def extract_and_crop_tables(self, implicit_rows: bool = False,
                                borderless_tables: bool = False,
                                min_confidence: int = 50,
                                save_path: str = None) -> List[np.ndarray]:
        """Extract tables from the image, crop them, and optionally save them to a folder."""
        # Invoke table extraction process
        return self.image.extract_and_crop_tables(implicit_rows=implicit_rows,
                                               borderless_tables=borderless_tables,
                                               min_confidence=min_confidence,
                                               save_path=save_path)