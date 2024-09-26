from setuptools import setup, find_packages

setup(
    name='pdf2table',
    version='0.1.4',
    author='rngzhi',
    description='pdf2table is a powerful Python tool designed to streamline the extraction of tabular data from PDF documents.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/li-rongzhi/pdf2table',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
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
        'poppler-utils'
    ],
)

