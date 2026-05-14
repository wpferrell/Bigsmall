"""BigSmall package setup."""
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bigsmall",
    version="1.0.0",
    description="Lossless neural network weight compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="BigSmall contributors",
    author_email="wpferrell@gmail.com",
    url="https://github.com/wpferrell/bigsmall",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(include=["bigsmall", "bigsmall.*"]),
    install_requires=[
        "numpy>=1.24",
        "constriction>=0.4",
        "zstandard>=0.21",
        "blosc2>=2.0",
        "safetensors>=0.4",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "hf": ["transformers>=4.30"],
        "diffusion": ["diffusers>=0.20"],
        "vllm": ["vllm>=0.4"],
        "all": [
            "torch>=2.0", "transformers>=4.30", "diffusers>=0.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "bigsmall=bigsmall.cli:main",
        ],
    },
)
