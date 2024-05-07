import copy
from functools import cached_property
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from pdf2table.table.processing.bordered_tables.cells import get_cells
from pdf2table.table.processing.bordered_tables.line import detect_lines
from pdf2table.table.processing.bordered_tables.tables import get_tables
from pdf2table.table.processing.bordered_tables.tables.implicit_rows import handle_implicit_rows
from pdf2table.table.metrics import compute_img_metrics
from pdf2table.table.processing.borderless_tables import identify_borderless_tables
from pdf2table.table.structure.models import Line
from pdf2table.table.structure.table_object import Table

from .utils import threshold_dark_areas

from .structure import TableObject
# from .utils import preprocess_image, detect_lines, detect_tables

@dataclass
class TableImage:
    img_array: np.ndarray
    min_confidence: int = 50
    char_length: Optional[float] = None
    median_line_sep: Optional[float] = None
    thresh: Optional[np.ndarray] = None
    contours: List[any] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    grayscale_img: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize the image processing."""
        self.convert_to_grayscale()
        self.char_length, self.median_line_sep, self.contours = compute_img_metrics(self.grayscale_img)

    def convert_to_grayscale(self):
        """Convert the image to grayscale if it is not already."""
        if self.img_array.ndim == 3 and self.img_array.shape[2] == 3:  # Check for color image
            self.grayscale_img = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2GRAY)
        elif self.img_array.ndim == 2:  # Already grayscale
            self.grayscale_img = self.img_array
        else:
            raise ValueError("Unsupported image array format: Image must be either grayscale or BGR color.")

    @cached_property
    def white_img(self) -> np.ndarray:
        white_img = copy.deepcopy(self.grayscale_img)

        # Draw white rows on detected rows
        for line in self.lines:
            if line.horizontal:
                cv2.rectangle(white_img, (line.x1 - line.thickness, line.y1), (line.x2 + line.thickness, line.y2),
                              (255, 255, 255), 3 * line.thickness)
            elif line.vertical:
                cv2.rectangle(white_img, (line.x1, line.y1 - line.thickness), (line.x2, line.y2 + line.thickness),
                              (255, 255, 255), 2 * line.thickness)
        return white_img

    def compute_image_metrics(self):
        """Compute metrics such as character length and median line separation."""
        self.char_length, self.median_line_sep, self.contours = detect_lines(self.thresh)

    def extract_tables(self, implicit_rows: bool = False, borderless_tables: bool = False, min_confidence: int=50):
        """Extract tables from the image with options for processing types."""
        self.extract_bordered_tables(implicit_rows)
        if borderless_tables:
            self.extract_borderless_tables()
        return self.tables

    def extract_bordered_tables(self, implicit_rows: bool):
        """Detect and extract bordered tables."""
        # Apply thresholding
        self.thresh = threshold_dark_areas(img=self.grayscale_img, char_length=self.char_length)

        # Compute parameters for line detection
        min_line_length = max(int(round(0.66 * self.median_line_sep)), 1) if self.median_line_sep else 20

        # Detect rows in image
        h_lines, v_lines = detect_lines(thresh=self.thresh,
                                        contours=self.contours,
                                        char_length=self.char_length,
                                        min_line_length=min_line_length)
        self.lines = h_lines + v_lines

        # Create cells from rows
        cells = get_cells(horizontal_lines=h_lines,
                          vertical_lines=v_lines)

        # Create tables from rows
        self.tables = get_tables(cells=cells,
                                 elements=self.contours,
                                 lines=self.lines,
                                 char_length=self.char_length)

        # If necessary, detect implicit rows
        if implicit_rows:
            self.tables = handle_implicit_rows(img=self.white_img,
                                               tables=self.tables,
                                               contours=self.contours)

        self.tables = [tb for tb in self.tables if tb.nb_rows * tb.nb_columns >= 4]

    def extract_borderless_tables(self):
        """Detect and extract borderless tables."""
        if self.median_line_sep is not None:
            # Extract borderless tables
            borderless_tbs = identify_borderless_tables(img=self.grayscale_img,
                                                        char_length=self.char_length,
                                                        median_line_sep=self.median_line_sep,
                                                        lines=self.lines,
                                                        contours=self.contours,
                                                        existing_tables=self.tables)

            # Add to tables
            self.tables += [tb for tb in borderless_tbs if tb.nb_rows >= 2 and tb.nb_columns >= 3]
