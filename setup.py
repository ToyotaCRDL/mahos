from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "mahos.ext.cqdyne_analyzer",
        ["mahos/ext/cqdyne_analyzer.cc"],
    ),
]

setup(ext_modules=ext_modules)
