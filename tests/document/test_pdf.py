import os
from tempfile import TemporaryDirectory

from pdf2table.document.pdf import PDF

def test_pdf_init():
    with TemporaryDirectory() as tmpdirname:
        test_pdf_path = os.path.join(tmpdirname, 'test.pdf')
        with open(test_pdf_path, 'w') as f:
            f.write('PDF content')

        pdf = PDF(test_pdf_path)
        assert pdf.pdf_path == test_pdf_path
        assert isinstance(pdf.pages, list)

# Test loading pages with non-existent file
def test_load_pages_nonexistent_file():
    pdf = PDF('dummy_file.pdf')
    assert pdf.load_pages() == []