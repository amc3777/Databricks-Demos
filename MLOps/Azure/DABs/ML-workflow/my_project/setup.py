"""
Setup script for my_project.

This script packages and distributes the associated wheel file(s).
Source code is in ./src/. Run 'python setup.py sdist bdist_wheel' to build.
"""
from setuptools import setup, find_packages

import sys
sys.path.append('./src')

import my_project

setup(
    name="my_project",
    version=my_project.__version__,
    url="https://databricks.com",
    author="andrew.cooley@databricks.com",
    description="my test wheel",
    packages=find_packages(where='./src'),
    package_dir={'': 'src'},
    entry_points={"entry_points": "main=my_project.main:main"},
    install_requires=["setuptools"],
)
