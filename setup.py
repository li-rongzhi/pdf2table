from setuptools import setup, find_packages

setup(
    name='pdf2table',
    version='0.1.0',
    author='rngzhi',
    description='pdf2table is a powerful Python tool designed to streamline the extraction of tabular data from PDF documents.',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'polars',
        'opencv-contrib-python',
        'numba',
        'numpy',
        'easyocr',
        'pdf2image',
        'torch',
        'transformers',
        'pandas',
        'pypdf',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
