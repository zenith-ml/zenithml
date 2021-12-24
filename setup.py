from __future__ import absolute_import, division, print_function

import os
import sys

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


# Find version
for line in open(os.path.join(PROJECT_PATH, "condorml", "__init__.py")):
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
    name="condorml",
    version=version,
    description="Pyro PPL on NumPy",
    packages=find_packages(include=["condorml", "condorml.*"]),
    url="https://github.com/condorml/condor",
    author="Uber AI Labs",
    install_requires=[
        "ray[all]",
        "fire==0.4.0",
        "rich==10.13.0",
        "gcsfs>=2021.7.0,<=2021.11.0",
        "scikit-learn==1.0.1",
        "nvtabular @ git+git://github.com/NVIDIA-Merlin/NVTabular.git@dc9c04d126b8ad842a32a3618e8f4495afbe0678",
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
            "check-manifest==0.47",
            "bump2version==1.0.1",
            "coverage==6.1.2",
            "flake8==4.0.1",
            "pytest==6.2.5",
            "pytest-cov==3.0.0",
            "pytest-mock",
            "jupyter==1.0.0",
            "jupyterlab==3.2.3",
            "mkdocs==1.2.3",
            "black==21.6b0",
            "toml-sort==0.19.0",
        ],
        "tf": [
            "tensorflow",
        ],
        "torch": [
            "torch==1.9.0",
        ],
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="GPU-Based ML Tools",
    license="Apache License 2.0",
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
