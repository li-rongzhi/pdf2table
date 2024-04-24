# coding: utf-8
from typing import List, Optional

from pdf2table.table.processing.borderless_tables.model import ColumnGroup
from pdf2table.table.processing.borderless_tables.table.coherency import check_table_coherency
from pdf2table.table.processing.borderless_tables.table.table_creation import get_table
from pdf2table.table.structure.table_object import Cell, Table



def identify_table(columns: ColumnGroup, row_delimiters: List[Cell], contours: List[Cell], median_line_sep: float,
                   char_length: float) -> Optional[Table]:
    """
    Identify table from column delimiters and rows
    :param columns: column delimiters group
    :param row_delimiters: list of table row delimitres corresponding to columns
    :param contours: list of image contours
    :param median_line_sep: median line separation
    :param char_length: average character length
    :return: Table object
    """
    # Create table from rows and columns delimiters
    table = get_table(columns=columns,
                      row_delimiters=row_delimiters,
                      contours=contours)

    if table:
        if check_table_coherency(table=table,
                                 median_line_sep=median_line_sep,
                                 char_length=char_length):
            return table

    return None
