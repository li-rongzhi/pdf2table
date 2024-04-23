from dataclasses import dataclass, field
import math
from typing import Optional, List

import numpy as np

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class TableCell:
    bbox: BBox
    value: Optional[str]

    def __hash__(self):
        return hash((self.bbox, self.value))

from collections import namedtuple
CellPosition = namedtuple('CellPosition', ['cell', 'row', 'col'])

@dataclass
class CellSpan:
    top_row: int
    bottom_row: int
    col_left: int
    col_right: int
    value: Optional[str]

    @property
    def colspan(self) -> int:
        return self.col_right - self.col_left + 1

    @property
    def rowspan(self) -> int:
        return self.bottom_row - self.top_row + 1

    @property
    def html_value(self) -> str:
        return self.value.replace("\n", "<br>") if self.value else ""

@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: Optional[int] = None
    angle: float = field(init=False)
    length: float = field(init=False)
    horizontal: bool = field(init=False)
    vertical: bool = field(init=False)

    def __post_init__(self):
        self.recalculate_properties()

    def recalculate_properties(self):
        """ Recalculate derived properties such as angle and length. """
        delta_x = self.x2 - self.x1
        delta_y = self.y2 - self.y1

        self.angle = math.atan2(delta_y, delta_x) * 180 / math.pi
        self.length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        self.horizontal = abs(self.angle) % 180 < 5  # Adjusted to handle near-horizontal lines
        self.vertical = abs(self.angle - 90) % 180 < 5  # Adjusted to handle near-vertical lines

    @property
    def dict(self):
        """ Return dictionary representation of the line. """
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "width": self.width, "height": self.height,
            "thickness": self.thickness
        }

    @property
    def transpose(self) -> "Line":
        """ Return a transposed version of this line. """
        return Line(x1=self.y1, y1=self.x1, x2=self.y2, y2=self.x2, thickness=self.thickness)

    def reprocess(self):
        """ Normalize line coordinates and adjust nearly horizontal or vertical lines. """
        self.x1, self.x2 = sorted([self.x1, self.x2])
        self.y1, self.y2 = sorted([self.y1, self.y2])

        self.recalculate_properties()  # Recalculate after coordinate normalization

    @property
    def width(self) -> float:
        """ Return the horizontal distance of the line. """
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        """ Return the vertical distance of the line. """
        return abs(self.y2 - self.y1)