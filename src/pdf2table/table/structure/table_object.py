from functools import cached_property
from dataclasses import dataclass

import numpy as np
import pandas as pd
from .models import TableCell, BBox
from typing import Optional, OrderedDict, Union, List
from dataclasses import dataclass, field
import copy

class TableObject:
    """A super class for all table related elements."""
    @cached_property
    def height(self) -> int:
        """Calculate and cache the height of the table object."""
        return self.y2 - self.y1

    @cached_property
    def width(self) -> int:
        """Calculate and cache the width of the table object."""
        return self.x2 - self.x1

    @cached_property
    def area(self) -> int:
        """Calculate and cache the area of the table object."""
        return self.height * self.width

    def bbox(self, margin: int = 0, height_margin: int = 0, width_margin: int = 0) -> tuple:
        """
        Return bounding box corresponding to the object, adjusted by specified margins.

        Args:
            margin (int): General margin applied uniformly around the object.
            height_margin (int): Additional vertical margin.
            width_margin (int): Additional horizontal margin.

        Returns:
            tuple: A tuple representing the bounding box (x1, y1, x2, y2).
        """
        # Calculate margins considering the general and specific margins
        x1_margin = margin if margin != 0 else width_margin
        y1_margin = margin if margin != 0 else height_margin
        x2_margin = margin if margin != 0 else width_margin
        y2_margin = margin if margin != 0 else height_margin

        return (self.x1 - x1_margin, self.y1 - y1_margin, self.x2 + x2_margin, self.y2 + y2_margin)

@dataclass
class Cell(TableObject):
    x1: int
    y1: int
    x2: int
    y2: int
    content: str = None

    @property
    def table_cell(self) -> TableCell:
        """Creates a TableCell instance from the Cell data."""
        return TableCell(bbox=BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2), value=self.content)

    def __hash__(self):
        """Generates a hash based on the cell's bounding box coordinates and content."""
        return hash((self.x1, self.y1, self.x2, self.y2, self.content))


class Row(TableObject):
    def __init__(self, cells: Union[Cell, List[Cell]]):
        if cells is None:
            raise ValueError("cells parameter is null")
        elif isinstance(cells, Cell):
            self._items = [cells]
        else:
            self._items = cells
        self._contours = []

    @property
    def items(self) -> List[Cell]:
        return self._items

    @property
    def nb_columns(self) -> int:
        return len(self.items)

    @property
    def x1(self) -> int:
        return min(map(lambda x: x.x1, self.items))

    @property
    def x2(self) -> int:
        return max(map(lambda x: x.x2, self.items))

    @property
    def y1(self) -> int:
        return min(map(lambda x: x.y1, self.items))

    @property
    def y2(self) -> int:
        return max(map(lambda x: x.y2, self.items))

    @property
    def v_consistent(self) -> bool:
        """
        Indicate if the row is vertically consistent (i.e all cells in row have the same vertical position)
        :return: boolean indicating if the row is vertically consistent
        """
        return all(map(lambda x: (x.y1 == self.y1) and (x.y2 == self.y2), self.items))

    def add_cells(self, cells: Union[Cell, List[Cell]]) -> "Row":
        """
        Add cells to existing row items
        :param cells: Cell object or list
        :return: Row object with cells added
        """
        if isinstance(cells, Cell):
            self._items += [cells]
        else:
            self._items += cells

        return self

    def split_in_rows(self, vertical_delimiters: List[int]) -> List["Row"]:
        """
        Split Row object into multiple objects based on vertical delimiters values
        :param vertical_delimiters: list of vertical delimiters values
        :return: list of splitted Row objects according to delimiters
        """
        # Create list of tuples for vertical boundaries
        row_delimiters = [self.y1] + vertical_delimiters + [self.y2]
        row_boundaries = [(i, j) for i, j in zip(row_delimiters, row_delimiters[1:])]

        # Create new list of rows
        l_new_rows = list()
        for boundary in row_boundaries:
            cells = list()
            for cell in self.items:
                _cell = copy.deepcopy(cell)
                _cell.y1, _cell.y2 = boundary
                cells.append(_cell)
            l_new_rows.append(Row(cells=cells))

        return l_new_rows

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            try:
                assert self.items == other.items
                return True
            except AssertionError:
                return False
        return False


@dataclass
class ExtractedTable:
    bbox: BBox
    title: Optional[str]
    content: OrderedDict[int, List[TableCell]]

    @property
    def df(self) -> pd.DataFrame:
        """
        Create pandas DataFrame representation of the table
        :return: pandas DataFrame containing table data
        """
        values = [[cell.value for cell in row] for k, row in self.content.items()]
        return pd.DataFrame(values)

class Table(TableObject):
    def __init__(self, rows: Union[Row, List[Row]], borderless: bool = False):
        if rows is None:
            self._items = []
        elif isinstance(rows, Row):
            self._items = [rows]
        else:
            self._items = rows
        self._title = None
        self._borderless = borderless

    @property
    def items(self) -> List[Row]:
        return self._items

    @property
    def title(self) -> str:
        return self._title

    def set_title(self, title: str):
        self._title = title

    @property
    def nb_rows(self) -> int:
        return len(self.items)

    @property
    def nb_columns(self) -> int:
        return self.items[0].nb_columns if self.items else 0

    @property
    def x1(self) -> int:
        return min(map(lambda x: x.x1, self.items))

    @property
    def x2(self) -> int:
        return max(map(lambda x: x.x2, self.items))

    @property
    def y1(self) -> int:
        return min(map(lambda x: x.y1, self.items))

    @property
    def y2(self) -> int:
        return max(map(lambda x: x.y2, self.items))

    @property
    def cell(self) -> Cell:
        return Cell(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def remove_rows(self, row_ids: List[int]):
        """
        Remove rows by ids
        :param row_ids: list of row ids to be removed
        """
        # Get remaining rows
        remaining_rows = [idx for idx in range(self.nb_rows) if idx not in row_ids]

        if len(remaining_rows) > 1:
            # Check created gaps between rows
            gaps = [(id_row, id_next) for id_row, id_next in zip(remaining_rows, remaining_rows[1:])
                    if id_next - id_row > 1]

            for id_row, id_next in gaps:
                # Normalize y value between rows
                y_gap = int(round((self.items[id_row].y2 + self.items[id_next].y1) / 2))

                # Put y value in both rows
                for c in self.items[id_row].items:
                    setattr(c, "y2", max(c.y2, y_gap))
                for c in self.items[id_next].items:
                    setattr(c, "y1", min(c.y1, y_gap))

        # Remove rows
        for idx in reversed(row_ids):
            self.items.pop(idx)

    def remove_columns(self, col_ids: List[int]):
        """
        Remove columns by ids
        :param col_ids: list of column ids to be removed
        """
        # Get remaining cols
        remaining_cols = [idx for idx in range(self.nb_columns) if idx not in col_ids]

        if len(remaining_cols) > 1:
            # Check created gaps between columns
            gaps = [(id_col, id_next) for id_col, id_next in zip(remaining_cols, remaining_cols[1:])
                    if id_next - id_col > 1]

            for id_col, id_next in gaps:
                # Normalize x value between columns
                x_gap = int(round(np.mean([row.items[id_col].x2 + row.items[id_next].x1 for row in self.items]) / 2))

                # Put x value in both columns
                for row in self.items:
                    setattr(row.items[id_col], "x2", max(row.items[id_col].x2, x_gap))
                    setattr(row.items[id_next], "x1", min(row.items[id_next].x1, x_gap))

        # Remove columns
        for idx in reversed(col_ids):
            for id_row in range(self.nb_rows):
                self.items[id_row].items.pop(idx)

    # def get_content(self, ocr_df: "OCRDataframe", min_confidence: int = 50) -> "Table":
    #     """
    #     Retrieve text from OCRDataframe object and reprocess table to remove empty rows / columns
    #     :param ocr_df: OCRDataframe object
    #     :param min_confidence: minimum confidence in order to include a word, from 0 (worst) to 99 (best)
    #     :return: Table object with data attribute containing dataframe
    #     """
    #     # Get content for each cell
    #     self = ocr_df.get_text_table(table=self, min_confidence=min_confidence)

    #     # Check for empty rows and remove if necessary
    #     empty_rows = list()
    #     for idx, row in enumerate(self.items):
    #         if all(map(lambda c: c.content is None, row.items)):
    #             empty_rows.append(idx)
    #     self.remove_rows(row_ids=empty_rows)

    #     # Check for empty columns and remove if necessary
    #     empty_cols = list()
    #     for idx in range(self.nb_columns):
    #         col_cells = [row.items[idx] for row in self.items]
    #         if all(map(lambda c: c.content is None, col_cells)):
    #             empty_cols.append(idx)
    #     self.remove_columns(col_ids=empty_cols)

    #     # Check for uniqueness of content
    #     unique_cells = set([cell for row in self.items for cell in row.items])
    #     if len(unique_cells) == 1:
    #         self._items = [Row(cells=self.items[0].items[0])]

    #     return self

    @property
    def extracted_table(self) -> ExtractedTable:
        bbox = BBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
        content = OrderedDict({idx: [cell.table_cell for cell in row.items] for idx, row in enumerate(self.items)})
        return ExtractedTable(bbox=bbox, title=self.title, content=content)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            try:
                assert self.items == other.items
                if self.title is not None:
                    assert self.title == other.title
                else:
                    assert other.title is None
                return True
            except AssertionError:
                return False
        return False