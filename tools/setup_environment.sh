#!/bin/bash

################################################################################
# Environment Setup Script
# Sets up the complete development and production environment using Conda
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "=========================================="
echo "   PPO Trading Agent - Environment Setup"
echo "=========================================="
echo -e "${NC}\n"

# Check for conda
echo -e "${YELLOW}[1/7] Checking for Conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda is not installed.${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
conda_version=$(conda --version 2>&1)
echo "Found: $conda_version"
echo -e "${GREEN}✓ Conda is available${NC}\n"

# Create conda environment
echo -e "${YELLOW}[2/7] Creating Conda environment...${NC}"
if conda env list | grep -q "ppo-trading"; then
    echo "Conda environment 'ppo-trading' already exists"
    echo "To recreate, run: conda env remove -n ppo-trading && bash tools/setup_environment.sh"
else
    conda env create -f environment.yml
    echo -e "${GREEN}✓ Conda environment created${NC}"
fi
echo

# Activate environment
echo -e "${YELLOW}[3/7] Activating Conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate ppo-trading
echo -e "${GREEN}✓ Conda environment activated${NC}\n"

# Verify GPU support
echo -e "${YELLOW}[4/7] Checking GPU/CUDA availability...${NC}"
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print('✓ GPU training is available')
else:
    print('No GPU detected - training will use CPU')
    print('(This is OK for development, GPU recommended for full training)')
"
echo

# Install package in development mode
echo -e "${YELLOW}[5/7] Installing package in development mode...${NC}"
pip install -e ".[dev]" --quiet
echo -e "${GREEN}✓ Package installed${NC}\n"

# Create directories
echo -e "${YELLOW}[6/7] Creating project directories...${NC}"
mkdir -p data models logs results .cache deployment/state reports monitoring/{prometheus,grafana/provisioning}
echo -e "${GREEN}✓ Directories created${NC}\n"

# Display information
echo -e "${YELLOW}[7/7] Setup Summary${NC}"
echo "=========================================="
echo -e "${GREEN}✓ Environment Setup Complete!${NC}"
echo "=========================================="
echo
echo "Next steps:"
echo "  1. Activate environment: conda activate ppo-trading"
echo "  2. Download data: python scripts/download_data.py --symbol AAPL"
echo "  3. Train model: python scripts/train.py"
echo "  4. Run tests: bash tools/run_tests.sh"
echo "  5. Deploy: python scripts/deploy.py --simulation"
echo
echo "For Docker deployment:"
echo "  docker-compose up -d"
echo
echo "Configuration files in: config/"
echo "Logs in: logs/"
echo "Models in: models/"
echo "Data in: data/"
echo

echo -e "${GREEN}Setup successful!${NC}\n"
