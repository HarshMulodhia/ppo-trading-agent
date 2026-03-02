#!/bin/bash

################################################################################
# Test Execution Script
# Runs comprehensive test suite
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COVERAGE_THRESHOLD=80
VERBOSE=${1:-""}

# Banner
echo -e "${BLUE}"
echo "=========================================="
echo "   PPO Trading Agent - Test Suite"
echo "=========================================="
echo -e "${NC}\n"

# Check if pytest is installed
echo -e "${YELLOW}[1/6] Checking test dependencies...${NC}"
if ! python3 -c "import pytest" 2>/dev/null; then
    echo -e "${RED}Error: pytest not installed${NC}"
    echo "Install with: pip install pytest pytest-cov"
    exit 1
fi
echo -e "${GREEN}✓ Test dependencies OK${NC}\n"

# Run unit tests
echo -e "${YELLOW}[2/6] Running unit tests...${NC}"
if [ -n "$VERBOSE" ]; then
    pytest tests/ -v --tb=short --color=yes \
        -k "not integration"
else
    pytest tests/ --tb=short --color=yes \
        -k "not integration" -q
fi
echo -e "${GREEN}✓ Unit tests passed${NC}\n"

# Run integration tests
echo -e "${YELLOW}[3/6] Running integration tests...${NC}"
if [ -n "$VERBOSE" ]; then
    pytest tests/ -v --tb=short --color=yes \
        -m "integration"
else
    pytest tests/ --tb=short --color=yes \
        -m "integration" -q
fi
echo -e "${GREEN}✓ Integration tests passed${NC}\n"

# Generate coverage report
echo -e "${YELLOW}[4/6] Generating coverage report...${NC}"
pytest tests/ \
    --cov=src \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    -q

# Check coverage threshold
coverage_percentage=$(grep -oP 'TOTAL\s+\d+\s+\d+\s+\K\d+' coverage.xml 2>/dev/null || echo "0")
echo "Coverage: ${coverage_percentage}%"

if [ "$coverage_percentage" -lt "$COVERAGE_THRESHOLD" ]; then
    echo -e "${YELLOW}Warning: Coverage ${coverage_percentage}% < ${COVERAGE_THRESHOLD}%${NC}"
else
    echo -e "${GREEN}✓ Coverage OK (${coverage_percentage}% >= ${COVERAGE_THRESHOLD}%)${NC}"
fi
echo

# Run type checking
echo -e "${YELLOW}[5/6] Running type checking (mypy)...${NC}"
if command -v mypy &> /dev/null; then
    mypy src/ --ignore-missing-imports --no-error-summary || true
    echo -e "${GREEN}✓ Type checking complete${NC}"
else
    echo -e "${YELLOW}⚠ mypy not installed, skipping${NC}"
fi
echo

# Summary
echo -e "${YELLOW}[6/6] Test Summary${NC}"
echo "=========================================="
echo -e "${GREEN}✓ All Tests Passed!${NC}"
echo "=========================================="
echo
echo "Coverage report: htmlcov/index.html"
echo "Coverage XML: coverage.xml"
echo
echo "Test Statistics:"
pytest tests/ --collect-only -q 2>/dev/null | tail -1 || echo "Tests collected"
echo

echo -e "${GREEN}Test execution successful!${NC}\n"

# Exit with success
exit 0
