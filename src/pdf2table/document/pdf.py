import os
from typing import List, Optional, Dict
from pdf2image import convert_from_path
from pypdf import PdfReader

from pdf2table.table.structure import TableObject

from .page import Page

class PDF:
    """
    A class to manage the operations related to a PDF file including loading pages,
    saving images conditionally, and extracting tables from pages.

    Attributes:
        pdf_path (str): The file path to the PDF document.
        pages (list): A list of Page objects representing each page of the PDF.
        length (int): The number of pages in the PDF.
    """

    def __init__(self, pdf_path: str, save_pages: bool = False) -> None:
        """
        Initializes the PDF object and loads the pages from the specified PDF file.

        Args:
            pdf_path (str): The path to the PDF file to be processed.
        """
        self.pdf_path = pdf_path
        self.pages = self.load_pages(save_pages=save_pages)
        self.length = len(self.pages)

    def load_pages(self, save_pages: bool = False) -> List[Page]:
        """
        Loads pages from a PDF file into Page objects and optionally saves them to a directory.

        Args:
            save_pages (bool): If True, saves each page as an image in a temporary directory.

        Returns:
            list: A list of Page objects.
        """
        if not os.path.isfile(self.pdf_path):
            print(f"File not found: {self.pdf_path}")
            return []

        pages = []
        output_dir = None
        base_name = os.path.splitext(os.path.basename(self.pdf_path))[0]

        if save_pages:
            output_dir = f"{base_name}_images"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        try:
            pdf_reader = PdfReader(self.pdf_path, strict=False)
            images = convert_from_path(self.pdf_path, fmt='jpeg', last_page=len(pdf_reader.pages))

            for page_number, image in enumerate(images, start=1):
                page = Page(image_data=image, page_number=page_number, metadata={'filename': f"{base_name}_page_{page_number}.jpg"})
                pages.append(page)

                if save_pages:
                    image_path = os.path.join(output_dir, page.metadata['filename'])
                    page.save_image(image_path)
                    print(f"Saved {image_path}")

        except Exception as e:
            print(f"Error reading PDF file: {e}")

        return pages

    def get_pages(self, page_numbers: List[int]) -> List[Page]:
        """
        Retrieves specific pages by their numbers.

        Args:
            page_numbers (list of int): A list of page numbers to retrieve.

        Returns:
            list: A list of Page objects corresponding to the requested page numbers.
        """
        pages = []
        for num in page_numbers:
            if 0 <= num - 1 < self.length:
                pages.append(self.pages[num - 1])
            else:
                print(f"Page number {num} out of range.")
        return pages

    def extract_tables(self, page_numbers: Optional[List[int]] = None,
                       implicit_rows: bool = False,
                       borderless_tables: bool = False,
                       min_confidence: int = 50) -> Dict[int, TableObject]:
        """
        Extracts tables from specified pages or all pages if no specific pages are provided.

        Args:
            page_numbers (list of int, optional): A list of page numbers from which to extract tables.

        Returns:
            dict: A dictionary where keys are page numbers and values are the extracted table data from each page.
        """
        if page_numbers is None:
            page_numbers = range(1, self.length + 1)

        extracted_tables = {}
        for num in page_numbers:
            page = self.get_pages([num])[0]
            if page:
                print(f"Extracting tables from page {num}")
                extracted_tables[num] = page.extract_tables(implicit_rows=implicit_rows, borderless_tables=borderless_tables,
                       min_confidence=min_confidence)
            else:
                print(f"Page number {num} is out of range or not valid")

        return extracted_tables

    def extract_and_crop_tables(self, page_numbers: Optional[List[int]] = None,
                       implicit_rows: bool = False,
                       borderless_tables: bool = False,
                       min_confidence: int = 50) -> Dict[int, TableObject]:
        if page_numbers is None:
            page_numbers = range(1, self.length + 1)

        extracted_tables = {}
        for num in page_numbers:
            page = self.get_pages([num])[0]
            if page:
                print(f"Extracting tables from page {num}")
                extracted_tables[num] = page.extract_and_crop_tables(implicit_rows=implicit_rows, borderless_tables=borderless_tables,
                       min_confidence=min_confidence)
            else:
                print(f"Page number {num} is out of range or not valid")

        return extracted_tables