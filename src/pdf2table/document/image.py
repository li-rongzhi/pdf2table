from typing import List
import numpy as np
from PIL import Image as Img
from dataclasses import dataclass
import os

from pdf2table.table import TableImage
from pdf2table.table.structure import TableObject

@dataclass
class Image:
    image_array: np.ndarray = None
    path: str = None

    def __post_init__(self):
        """Load image from path or convert provided PIL Image to array upon initialization."""
        if self.path and not self.image_array:
            if os.path.exists(self.path):
                self.load_image_from_path(self.path)
            else:
                raise FileNotFoundError(f"No file found at {self.path}")
        elif not self.path and isinstance(self.image_array, Img.Image):
            self.convert_pil_to_array(self.image_array)
        elif not self.path and not isinstance(self.image_array, (np.ndarray, type(None))):
            raise TypeError("image_array must be a NumPy array or a PIL Image.")

    def load_image_from_path(self, path: str):
        """Load an image from a file path into a NumPy array."""
        image = Img.open(path)
        self.convert_pil_to_array(image)

    def convert_pil_to_array(self, image: Img.Image):
        """Convert a PIL Image to a NumPy array and store it."""
        self.image_array = np.array(image)

    def display_image(self):
        """Display the image using the default image viewer."""
        if self.image_array is not None:
            image = Img.fromarray(self.image_array)
            image.show()
        else:
            raise ValueError("No image data is available to display.")

    def save_image(self, path: str, format='JPEG'):
        """Save the image to a specified path."""
        if self.image_array is not None:
            image = Img.fromarray(self.image_array)
            format = format if format else 'JPEG'  # Default format if not specified
            image.save(path, format.upper())
        else:
            raise ValueError("No image data is available to save.")

    def extract_tables(self, implicit_rows: bool = False, borderless_tables: bool = False,
                       min_confidence: int = 50) -> List[TableObject]:
        """Extract tables from the page."""
        table_image = TableImage(img_array=self.image_array)
        return table_image.extract_tables(implicit_rows=implicit_rows, borderless_tables=borderless_tables,
                                         min_confidence=min_confidence)



