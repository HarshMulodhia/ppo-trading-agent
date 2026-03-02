#!/bin/bash

################################################################################
# Code Quality and Formatting Script
# Performs linting, formatting, and type checking
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FIX=${1:-""}  # Pass "fix" to auto-fix issues
SRC_DIR="src"
TESTS_DIR="tests"
SCRIPTS_DIR="scripts"

# Banner
echo -e "${BLUE}"
echo "=========================================="
echo "   PPO Trading Agent - Code Quality"
echo "=========================================="
echo -e "${NC}\n"

# Colors for tools
echo -e "${YELLOW}Configuration:${NC}"
echo "  Fix issues: ${FIX:-disabled}"
echo "  Source dir: $SRC_DIR"
echo "  Tests dir: $TESTS_DIR"
echo

# Check tools
echo -e "${YELLOW}[1/5] Checking code quality tools...${NC}"
tools_found=0

if command -v black &> /dev/null; then
    echo "  ✓ black (formatter)"
    ((++tools_found))
else
    echo "  ✗ black not installed"
fi

if command -v flake8 &> /dev/null; then
    echo "  ✓ flake8 (linter)"
    ((++tools_found))
else
    echo "  ✗ flake8 not installed"
fi

if command -v isort &> /dev/null; then
    echo "  ✓ isort (import sorter)"
    ((++tools_found))
else
    echo "  ✗ isort not installed"
fi

if command -v mypy &> /dev/null; then
    echo "  ✓ mypy (type checker)"
    ((++tools_found))
else
    echo "  ✗ mypy not installed"
fi

if [ "$tools_found" -eq 0 ]; then
    echo -e "${YELLOW}Install tools with: pip install black flake8 isort mypy pylint${NC}"
fi
echo

# Format code with black
echo -e "${YELLOW}[2/5] Formatting code with black...${NC}"
if command -v black &> /dev/null; then
    black "$SRC_DIR" "$TESTS_DIR" "$SCRIPTS_DIR" --line-length=100 --quiet
    echo -e "${GREEN}✓ Code formatted${NC}"
else
    echo -e "${YELLOW}⚠ black not installed, skipping${NC}"
fi
echo

# Sort imports with isort
echo -e "${YELLOW}[3/5] Sorting imports with isort...${NC}"
if command -v isort &> /dev/null; then
    isort "$SRC_DIR" "$TESTS_DIR" "$SCRIPTS_DIR" --quiet
    echo -e "${GREEN}✓ Imports sorted${NC}"
else
    echo -e "${YELLOW}⚠ isort not installed, skipping${NC}"
fi
echo

# Lint with flake8
echo -e "${YELLOW}[4/5] Linting with flake8...${NC}"
if command -v flake8 &> /dev/null; then
    echo "Checking $SRC_DIR..."
    if flake8 "$SRC_DIR" \
        --max-line-length=100 \
        --extend-ignore=E203,W503 \
        --count \
        --statistics; then
        echo -e "${GREEN}✓ Linting passed${NC}"
    else
        echo -e "${YELLOW}⚠ Linting issues found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ flake8 not installed, skipping${NC}"
fi
echo

# Type checking with mypy
echo -e "${YELLOW}[5/5] Type checking with mypy...${NC}"
if command -v mypy &> /dev/null; then
    if mypy "$SRC_DIR" --ignore-missing-imports --no-error-summary; then
        echo -e "${GREEN}✓ Type checking passed${NC}"
    else
        echo -e "${YELLOW}⚠ Type checking issues found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ mypy not installed, skipping${NC}"
fi
echo

# Summary
echo "=========================================="
echo -e "${GREEN}✓ Code Quality Check Complete!${NC}"
echo "=========================================="
echo
echo "Recommendations:"
echo "  1. Review any linting or type issues"
echo "  2. Fix import order if isort was skipped"
echo "  3. Run tests: bash tools/run_tests.sh"
echo "  4. Generate docs: bash tools/generate_docs.sh"
echo

echo -e "${GREEN}Code quality check successful!${NC}\n"
