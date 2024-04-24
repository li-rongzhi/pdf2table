import os

import torch
from transformers import AutoModelForObjectDetection
import easyocr

from pdf2table.document import PDF
from pdf2table.document import Image
from pdf2table.tatr import TATR

class Driver:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.structure_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all").to(self.device)
        self.reader = easyocr.Reader(['en'])
        self.tatr = TATR(self.structure_model, self.device, self.reader)

    def check_file_type(self, filepath):
        """Check the file extension to determine how to process the file."""
        _, file_extension = os.path.splitext(filepath)
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            return 'image'
        elif file_extension.lower() == '.pdf':
            return 'pdf'
        else:
            raise ValueError("Unsupported file type")

    def extract_tables(self, filepath, implicit_rows=False, borderless_tables=False, min_confidence=50):
        """Determine the file type and invoke the extract_tables method of the appropriate object."""
        filetype = self.check_file_type(filepath)
        if filetype == 'image':
            file_object = Image(filepath)
        elif filetype == 'pdf':
            file_object = PDF(filepath)
        else:
            raise ValueError("Unsupported file type")
        return file_object.extract_tables(implicit_rows=implicit_rows, borderless_tables=borderless_tables, min_confidence=min_confidence)