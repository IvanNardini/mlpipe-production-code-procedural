#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="procedural_ml_pipe_ivnard",
    version="0.0.1",
    author="Ivan Nardini",
    author_email="ivan.nardini@sas.com",
    description="A package for procedural ml pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IvanNardini/mlpipe-production-code-procedural.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)