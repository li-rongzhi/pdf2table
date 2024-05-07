# pdf2table
[![PyPI](https://img.shields.io/pypi/v/pdf2table.svg)](https://pypi.org/project/pdf2table/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pdf2table)](https://img.shields.io/pypi/dm/pdf2table)
![GitHub](https://img.shields.io/github/license/li-rongzhi/pdf2table.svg)

pdf2table is a Python library designed to extract tabular data from PDF files and images efficiently and accurately. It leverages an enhanced algorithm of `img2table` library for table detection and the TATR model from Microsoft's Table Transformer for precise table structure recognition and content extraction.

## Features
- **High Precision of Detection**: Compared to Table Transformer's DETR model, rule-based algorithm is less likely to identify text blocks as table regions.
- **Maintenance Structural Information**: Utilizes state-of-the-art models for table structure recognition to maintain structural information of tables.
- **Flexible Input**: Supports both PDF files and image formats for table extraction. (More file format will be available later)
- **Easy to Use**: Simple API allows for straightforward integration into Python projects.

## Installation

Install pdf2table using pip:

```bash
pip install pdf2table
```

## Usage
Here's a quick example on how to use PDF2Table to extract tables from a PDF file:
```python
from pdf2table import Driver

# Initialize the driver
driver = Driver()

# Extract tables from a PDF
# which returns a list of dataframes
tables = driver.extract_tables("sample.pdf")

```
`Driver` object encapsulates the detection and extraction for both `PDF` object and `Image` object. If detection is what you need, please refer to the following example:
```python
from pdf2table.document import Image, PDF

# Initialize an Image object
img = Image("sample.jpg")

# Extract all tables from the image
# which returns a list of Table objects
img_tables = img.extract_tables()

# Initialize an PDF object
pdf = PDF("sample.jpg")
pdf_tables = pdf.extract_tables()
```
You may refer to [tutorial](samples/tutorial.ipynb) for more details
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Thanks to the creators of the [img2table](https://github.com/xavctn/img2table) library and Microsoft's [Table Transformer](https://github.com/microsoft/table-transformer) model for providing the robust foundations for this tool.
