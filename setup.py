"""BigSmall package setup."""
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bigsmall",
    version="1.0.1",
    description="Lossless neural network weight compression - run any model, no compromises",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Will Ferrell",
    author_email="wpferrell@gmail.com",
    url="https://github.com/wpferrell/Bigsmall",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(include=["bigsmall", "bigsmall.*"]),
    install_requires=[
        "numpy>=1.24",
        "constriction>=0.4",
        "zstandard>=0.21",
        "blosc2>=2.0",
        "safetensors>=0.4",
        "huggingface-hub>=0.20",
        "tqdm>=4.0",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "hf": ["transformers>=4.30", "huggingface-hub>=0.20"],
        "diffusion": ["diffusers>=0.20"],
        "vllm": ["vllm>=0.4"],
        "all": [
            "torch>=2.0", "transformers>=4.30",
            "diffusers>=0.20", "huggingface-hub>=0.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "bigsmall=bigsmall.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Archiving :: Compression",
    ],
    keywords=["machine learning", "compression", "lossless", "neural networks", "LLM", "transformers"],
)
