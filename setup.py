from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vision-interpretability-toolkit",
    version="0.1.0",
    author="Your Name",
    description="A clean interpretability toolkit for ViT/ConvNet models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.12",
        "captum>=0.7.0",
        "vit-prisma>=0.1.2",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
