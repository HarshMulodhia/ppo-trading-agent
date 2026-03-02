#!/bin/bash

################################################################################
# Environment Setup Script
# Sets up the complete development and production environment
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

# Check Python version
echo -e "${YELLOW}[1/8] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}\n"

# Create virtual environment
echo -e "${YELLOW}[2/8] Creating virtual environment...${NC}"
if [ -d ".rl" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv .rl
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}[3/8] Activating virtual environment...${NC}"
source .rl/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}\n"

# Upgrade pip, setuptools, wheel
echo -e "${YELLOW}[4/8] Upgrading pip, setuptools, wheel...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip tools upgraded${NC}\n"

# Install requirements
echo -e "${YELLOW}[5/8] Installing Python dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi
echo

# Create directories
echo -e "${YELLOW}[6/8] Creating project directories...${NC}"
mkdir -p data models logs results .cache deployment/state reports monitoring/{prometheus,grafana/provisioning}
echo -e "${GREEN}✓ Directories created${NC}\n"

# Install development tools
echo -e "${YELLOW}[7/8] Installing development tools...${NC}"
pip install pytest pytest-cov black flake8 isort mypy
echo -e "${GREEN}✓ Development tools installed${NC}\n"

# Display information
echo -e "${YELLOW}[8/8] Setup Summary${NC}"
echo "=========================================="
echo -e "${GREEN}✓ Environment Setup Complete!${NC}"
echo "=========================================="
echo
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
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
