from setuptools import setup, find_packages

setup(
    name='pdf2table',
    version='0.1.0',
    author='rongzhi',
    description='pdf2table is a powerful Python tool designed to streamline the extraction of tabular data from PDF documents.',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
