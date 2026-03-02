# PPO Trading Agent - Single-Agent Adaptive Trading with Reinforcement Learning

A production-grade implementation of a trading agent using Proximal Policy Optimization (PPO), with comprehensive documentation, evaluation framework, and deployment tools.

## Overview

This project implements a **Single-Agent Adaptive Trading Agent** that learns optimal Buy/Sell/Hold decisions using PPO, a state-of-the-art policy gradient algorithm. The agent:

- Learns from historical OHLCV + technical indicator data
- Optimizes risk-adjusted returns (Sharpe ratio)
- Uses actor-critic architecture for sample efficiency
- Includes comprehensive backtesting framework
- Production-ready with risk management constraints

**Timeline:** 3-4 months | **Difficulty:** Beginner-Intermediate | **Status:** Production-Ready

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/HarshMulodhia/ppo-trading-agent.git
cd ppo-trading-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import gymnasium, stable_baselines3; print('✓ Ready!')"
```

### First Run

```bash
# Download data
python scripts/download_data.py --symbol NIFTY50 --start 2019-01-01 --end 2023-12-31

# Train agent
python scripts/train.py --config config/training_config.yaml

# Evaluate results
python scripts/evaluate.py --model models/baseline/model.zip
```

## Project Structure

```
ppo-trading-agent/
├── src/                    # Source code (environment, agent, training, evaluation)
├── scripts/                # Executable scripts (train, evaluate, tune)
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit and integration tests
├── docs/                   # Comprehensive documentation
├── config/                 # Configuration files (YAML)
├── data/                   # Data storage
├── models/                 # Trained model checkpoints
├── logs/                   # Training logs and metrics
└── outputs/                # Results and visualizations
```

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed directory breakdown.

## Documentation

| Document | Purpose |
|----------|---------|
| **[PPO_Trading_Guide.md](docs/PPO_Trading_Guide.md)** | Comprehensive 12-section implementation guide with theory and code |
| **[Quick_Reference.md](docs/Quick_Reference.md)** | Code templates, checklists, and troubleshooting |
| **[Executive_Summary.md](docs/Executive_Summary.md)** | Strategic overview and 4-month roadmap |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design and component relationships |
| **[API_REFERENCE.md](docs/API_REFERENCE.md)** | API documentation for all modules |

## Core Components

### Environment (Gymnasium)
- **State:** OHLCV + technical indicators (RSI, MACD, Bollinger Bands)
- **Actions:** Discrete {Buy, Sell, Hold}
- **Reward:** Composite Sharpe ratio + risk penalty
- **Features:** Customizable, validated, production-tested

### PPO Agent
- **Architecture:** Actor-Critic with shared backbone
- **Networks:** 2-layer MLPPolicy with LayerNorm
- **Optimization:** Adam with clipped surrogate objective
- **Stability:** Entropy regularization + gradient clipping

### Training Pipeline
- **Framework:** Stable-Baselines3 (PyTorch backend)
- **Callbacks:** Checkpointing, evaluation, early stopping
- **Monitoring:** TensorBoard integration + custom logging
- **Hyperparameters:** Configurable, with tuning support

### Evaluation Framework
- **Metrics:** Sharpe ratio, max drawdown, win rate, Calmar ratio
- **Backtesting:** Walk-forward with transaction costs
- **Baseline:** Buy-and-hold comparison
- **Visualization:** Performance curves, action distributions

## Key Features

✅ **Production-Ready**
- Risk management constraints
- Model versioning and deployment
- Concept drift monitoring
- Comprehensive error handling

✅ **Well-Documented**
- 12-section implementation guide
- Inline code comments
- Architecture diagrams
- API reference

✅ **Fully Tested**
- Unit tests for all components
- Integration tests for pipeline
- Environment validation
- Performance benchmarks

✅ **Extensible**
- Modular architecture
- Custom reward functions
- Multiple data sources
- Easy to extend to multi-asset

## Training Configuration

```yaml
# config/training_config.yaml
ppo:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5

training:
  total_timesteps: 500000
  eval_freq: 5000
  save_freq: 50000
```

## Performance Targets

**Baseline Expectations (Nifty50):**
- Sharpe Ratio: 0.8-1.2 (vs 0.4-0.6 buy-and-hold)
- Maximum Drawdown: 12-18% (vs 20-30%)
- Win Rate: 52-58%
- Calmar Ratio: 1.0-1.5

## Usage Examples

### Training

```python
from src.training.trainer import Trainer
from src.training.config import TrainingConfig

# Load configuration
config = TrainingConfig.from_yaml("config/training_config.yaml")

# Create trainer
trainer = Trainer(config)

# Train agent
model = trainer.train(
    total_timesteps=500000,
    eval_interval=5000
)

# Save model
model.save("models/trained_agent")
```

### Evaluation

```python
from src.agent.ppo_agent import PPOAgent
from src.evaluation.evaluator import Evaluator

# Load trained model
agent = PPOAgent.load("models/trained_agent")

# Evaluate on test set
evaluator = Evaluator(test_env)
metrics = evaluator.evaluate(agent, episodes=10)

print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.3f}")
print(f"Win Rate: {metrics['win_rate']:.3f}")
```

### Hyperparameter Tuning

```python
from src.training.tuner import HyperparameterTuner

tuner = HyperparameterTuner(
    n_trials=20,
    direction="maximize",
    metric="sharpe_ratio"
)

best_params = tuner.optimize()
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_environment.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Deployment

### Local Inference

```python
from src.deployment.inference import InferenceEngine

engine = InferenceEngine("models/trained_agent")

# Get action for current state
action = engine.predict(observation)
print(f"Recommended action: {['Hold', 'Buy', 'Sell'][action]}")
```

### Production Deployment

```bash
# Build Docker image
docker build -f docker/Dockerfile -t ppo-trading-agent .

# Run container
docker run -d --name trading-agent \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=models/trained_agent \
  ppo-trading-agent
```

## Roadmap

### Phase 1: Foundation (Month 1-2)
- ✅ Environment implementation
- ✅ PPO agent training
- ✅ Basic evaluation

### Phase 2: Optimization (Month 2-3)
- ✅ Hyperparameter tuning
- ✅ Reward function refinement
- ✅ Multi-asset extension

### Phase 3: Production (Month 3-4)
- ✅ Risk management
- ✅ Deployment pipeline
- ✅ Monitoring system

## Contributing

Contributions welcome! Areas for improvement:
- Additional RL algorithms (A3C, SAC)
- Multi-agent extensions
- Real-time trading integration
- Advanced reward shaping
- Performance optimization

## Advanced Topics

### Multi-Agent Extension (MARL)
Extend to portfolio of multiple agents learning cooperatively/competitively.

### Transfer Learning
Train on bull market, fine-tune on bear market with reduced data.

### Ensemble Methods
Combine predictions from multiple PPO agents for robustness.

### Real-Time Trading
Live market integration with proper safeguards and constraints.

## Troubleshooting

### Agent not learning
- Increase learning rate (1e-3)
- Increase entropy coefficient (0.05)
- Verify reward signal has variance

### Training unstable
- Decrease learning rate (1e-4)
- Decrease clip range (0.1)
- Increase number of epochs (15)

### Poor generalization
- Test on different time period
- Reduce model capacity
- Add regularization

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more issues.

## Performance Benchmarks

**Hardware:** GPU (Tesla V100), RAM 16GB  
**Training Time:** ~4 hours for 500k timesteps  
**Inference:** <1ms per decision  
**Model Size:** 2-5 MB

## References

- **Schulman et al. (2017):** "Proximal Policy Optimization Algorithms"
- **Stable Baselines3:** https://stable-baselines3.readthedocs.io
- **Gymnasium:** https://gymnasium.farama.org
- **OpenAI Spinning Up:** https://spinningup.openai.com

## Citation

```bibtex
@software{ppo_trading_agent_2025,
  title={Single-Agent Adaptive Trading Agent with PPO},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ppo-trading-agent}
}
```

## License

MIT License - See LICENSE file for details

## Support

- **Documentation:** See `/docs` folder
- **Issues:** GitHub Issues tab
- **Discussions:** GitHub Discussions
- **Email:** your.email@example.com

---

**Status:** Production-Ready | **Version:** 1.0 | **Last Updated:** December 2025