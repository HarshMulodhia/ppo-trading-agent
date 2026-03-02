"""
Setup script for PPO Trading Agent package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ppo-trading-agent",
    version="1.0.0",
    author="Harsh Mulodhia",
    author_email="hajiharsh598@gmail.com",
    description="Single-Agent Adaptive Trading Agent using Proximal Policy Optimization (PPO)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HarshMulodhia/ppo-trading-agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "stable-baselines3>=2.0.0",
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "yfinance>=0.2.28",
        "pyyaml>=6.0",
        "pyarrow>=12.0.0",
    ],
    extras_require={
        "financial": [
            "ta-lib>=0.4.24",
            "nsepy>=0.8",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "plotly>=5.14.0",
            "seaborn>=0.12.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "jupyter>=1.0.0",
        ],
        "optimization": [
            "optuna>=3.8.0",
            "scikit-learn>=1.2.0",
        ],
        "production": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "pydantic>=2.0.0",
        ],
        "monitoring": [
            "tensorboard>=2.12.0",
            "wandb>=0.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ppo-trading=scripts.train:main",
        ],
    },
    keywords="reinforcement learning trading PPO policy optimization finance",
    project_urls={
        "Documentation": "https://github.com/HarshMulodhia/ppo-trading-agent/tree/main/docs",
        "Source": "https://github.com/HarshMulodhia/ppo-trading-agent",
        "Tracker": "https://github.com/HarshMulodhia/ppo-trading-agent/issues",
    },
    zip_safe=False,
)