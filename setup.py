"""BigSmall package setup."""
from setuptools import setup, find_packages

setup(
    name="bigsmall",
    version="1.0.0",
    description="Lossless neural network weight compression",
    long_description=(
        "BigSmall is a lossless compressor for neural network weights. "
        "It supports FP32, BF16, FP16, FP8, and FP4 weights, with delta "
        "compression for fine-tuned models, vLLM and HuggingFace integration, "
        "and diffusion model support."
    ),
    author="BigSmall contributors",
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
