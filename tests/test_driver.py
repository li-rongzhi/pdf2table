import pytest
from pdf2table.driver import Driver

# Use the fixture decorator to create a setup fixture
@pytest.fixture
def driver():
    return Driver()

# Simply pass the fixture as a parameter to the test functions
def test_extract_tables_from_image(driver):
    filepath = "tests/test_data/test_image.jpg"
    tables = driver.extract_tables(filepath)
    assert isinstance(tables, list)
    assert all(isinstance(tb, dict) for tb in tables)

def test_extract_tables_from_pdf(driver):
    filepath = "tests/test_data/test_document.pdf"
    tables = driver.extract_tables(filepath)
    assert isinstance(tables, dict)
    assert all(isinstance(page_tables, list) for page_tables in tables.values())

def test_detect_tables_from_image(driver):
    filepath = "tests/test_data/test_image.jpg"
    tables = driver.detect_tables(filepath)
    assert isinstance(tables, list)
    assert all(isinstance(tb, dict) for tb in tables)

def test_detect_tables_from_pdf(driver):
    filepath = "tests/test_data/test_document.pdf"
    tables = driver.detect_tables(filepath)
    assert isinstance(tables, dict)
    assert all(isinstance(page_tables, list) for page_tables in tables.values())

