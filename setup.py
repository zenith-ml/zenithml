from __future__ import absolute_import, division, print_function

import os
import sys

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


# Find version
for line in open(os.path.join(PROJECT_PATH, "zenithml", "__init__.py")):
    if line.startswith("__version__ = "):
        version = line.strip().split()[2][1:-1]

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README.md:\n  {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

setup(
    name="zenithml",
    version=version,
    description="Praveen Chandar",
    packages=find_packages(include=["zenithml", "zenithml.*"]),
    url="https://github.com/zenith/zenith",
    author="Praveen Chandar",
    install_requires=[
        "cryptography~=3.1",
        "fire>=0.4.0",
        "gcsfs>=2021.7.0,<=2021.11.0",
        "google-api-python-client>=2.33.0",
        "google-cloud-bigquery>=2.0.0,<3.0.0",
        "ray[all]>=1.9.0",
        "ray[gcp]>=1.9.0",
        "rich>=10.13.0",
        "scikit-learn>=1.0.1",
        "sqlparse>=0.4.2",
        "nvtabular @ git+git://github.com/NVIDIA-Merlin/NVTabular.git@v0.11.0",
    ],
    extras_require={
        "docs": [
            "ipython",  # sphinx needs this to render codes
            "nbsphinx>=0.8.5",
            "readthedocs-sphinx-search==0.1.0",
            "sphinx",
            "sphinx_rtd_theme",
            "sphinx-gallery",
        ],
        "dev": [
            "black>=21.6b0",
            "bump2version>=1.0.1",
            "coverage>=6.1.2",
            "flake8>=4.0.1",
            "isort>=5.10.1",
            "jupyter>=1.0.0",
            "jupyterlab>=3.2.3",
            "pytest>=6.2.5",
            "pytest-cov>=3.0.0",
            "pytest-mock",
            "mypy>=0.920",
        ],
        "tf": [
            "tensorflow",
        ],
        "torch": [
            "torch",
        ],
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Machine Learning, ETL, Tensorflow, PyTorch",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
