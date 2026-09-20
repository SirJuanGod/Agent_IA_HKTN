from setuptools import setup, find_packages

setup(
    name="rbj",
    version="0.1.0",
    description="SJG-Agent — Cognitive AI Agent with Natural Language Interface",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
)
