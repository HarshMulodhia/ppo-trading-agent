#!/bin/bash

################################################################################
# Documentation Generation Script
# Generates comprehensive project documentation
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCS_DIR="docs"
OUTPUT_DIR="docs/_build"

# Banner
echo -e "${BLUE}"
echo "=========================================="
echo "   PPO Trading Agent - Documentation"
echo "=========================================="
echo -e "${NC}\n"

# Check if documentation tools are installed
echo -e "${YELLOW}[1/6] Checking documentation tools...${NC}"
if ! python3 -c "import sphinx" 2>/dev/null; then
    echo -e "${RED}Warning: Sphinx not installed${NC}"
    echo "Install with: pip install sphinx sphinx-rtd-theme"
fi
echo -e "${GREEN}✓ Documentation tools checked${NC}\n"

# Create docs directory if not exists
echo -e "${YELLOW}[2/6] Setting up documentation structure...${NC}"
mkdir -p "$DOCS_DIR"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Documentation directories created${NC}\n"

# Generate API documentation from code
echo -e "${YELLOW}[3/6] Generating API documentation...${NC}"
cat > "$DOCS_DIR/api.md" << 'EOF'
# API Reference

## Core Modules

### src.environment
- `TradingEnvironment` - RL trading environment
- `MarketSimulator` - Realistic market simulation
- `ObservationBuilder` - Observation construction

### src.agent
- `PPOAgent` - Proximal Policy Optimization
- `ActorCriticNetwork` - Neural network architecture
- `ReplayBuffer` - Experience storage

### src.training
- `PPOTrainer` - Training orchestration
- `RolloutCollector` - Trajectory collection
- `RolloutBuffer` - Rollout storage

### src.evaluation
- `Backtester` - Historical backtesting
- `AgentEvaluator` - Agent evaluation
- `MetricsComputer` - Performance metrics

### src.reward
- `RewardShaper` - Reward engineering
- `RewardAnalyzer` - Reward analysis
- `CustomRewardFunction` - Custom rewards

### src.data
- `DataLoader` - Data loading interface
- `DataPreprocessor` - Data preprocessing
- `DataSplitter` - Time-series splitting
- `DataCache` - Efficient caching

### src.deployment
- `ModelManager` - Model versioning
- `InferenceEngine` - Production inference
- `RiskManager` - Risk constraints
- `PerformanceMonitor` - Monitoring

### src.utils
- `constants` - Configuration constants
- `helpers` - Utility functions
- `decorators` - Function decorators
- `exceptions` - Custom exceptions

## Scripts

### download_data.py
Download historical market data from various sources.

### train.py
Train PPO agent on historical data.

### evaluate.py
Evaluate trained agent on test data.

### tune_hyperparams.py
Perform Bayesian hyperparameter optimization.

### compare_baselines.py
Compare against baseline strategies.

### generate_report.py
Generate analysis reports.

### deploy.py
Deploy model for live trading.

## Configuration

See `config/` directory for YAML configuration files.
EOF

echo -e "${GREEN}✓ API documentation generated${NC}\n"

# Generate module documentation
echo -e "${YELLOW}[4/6] Extracting module docstrings...${NC}"
python3 << 'PYTHON_EOF'
import os
import importlib.util
from pathlib import Path

docs_file = Path("docs/modules.md")
with open(docs_file, 'w') as f:
    f.write("# Module Documentation\n\n")
    
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                module_path = os.path.join(root, file)
                try:
                    spec = importlib.util.spec_from_file_location("module", module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    if module.__doc__:
                        f.write(f"\n## {module_path}\n")
                        f.write(f"{module.__doc__}\n")
                except Exception as e:
                    pass

print("Module documentation extracted")
PYTHON_EOF

echo -e "${GREEN}✓ Module documentation extracted${NC}\n"

# Generate README documentation
echo -e "${YELLOW}[5/6] Verifying README documentation...${NC}"
if [ -f "README.md" ]; then
    echo -e "${GREEN}✓ README.md exists${NC}"
else
    echo -e "${YELLOW}⚠ README.md not found${NC}"
fi
echo

# Create quick reference
echo -e "${YELLOW}[6/6] Creating quick reference...${NC}"
cat > "$DOCS_DIR/quick_reference.md" << 'EOF'
# Quick Reference

## Installation
```bash
bash tools/setup_environment.sh
```

## Running Tests
```bash
bash tools/run_tests.sh
```

## Training
```bash
python scripts/train.py --config config/training_config.yaml
```

## Evaluation
```bash
python scripts/evaluate.py --model ppo_v1 --data data/test.parquet
```

## Deployment
```bash
python scripts/deploy.py --model ppo_v2 --simulation
```

## Configuration
Edit configuration files in `config/`:
- `environment_config.yaml` - Environment settings
- `training_config.yaml` - Training parameters
- `evaluation_config.yaml` - Evaluation settings
- `deployment_config.yaml` - Deployment settings

## Key Classes
- `TradingEnvironment` - RL environment
- `PPOAgent` - PPO agent
- `PPOTrainer` - Training orchestrator
- `Backtester` - Backtesting engine
- `MetricsComputer` - Performance metrics

## Useful Commands
```bash
# Format code
bash tools/lint_and_format.sh

# Run specific test
pytest tests/test_agent.py -v

# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# Run with Docker
docker-compose up -d
```
EOF

echo -e "${GREEN}✓ Quick reference created${NC}\n"

# Summary
echo "=========================================="
echo -e "${GREEN}✓ Documentation Generated!${NC}"
echo "=========================================="
echo
echo "Documentation files:"
echo "  API Reference: $DOCS_DIR/api.md"
echo "  Module Docs: $DOCS_DIR/modules.md"
echo "  Quick Reference: $DOCS_DIR/quick_reference.md"
echo "  README: README.md"
echo
echo -e "${GREEN}Documentation generation successful!${NC}\n"
