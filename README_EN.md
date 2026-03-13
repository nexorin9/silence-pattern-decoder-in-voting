# Silence Pattern Decoder in Voting

Silence/Abstention Voting Pattern Decoder - A Social Science Experimental Computational Analysis Tool

[![Tests](https://github.com/your-org/silence-decoder/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/silence-decoder/actions/workflows/tests.yml)
[![Lint](https://github.com/your-org/silence-decoder/actions/workflows/lint.yml/badge.svg)](https://github.com/your-org/silence-decoder/actions/workflows/lint.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## Introduction

This is a **social science research tool** that uses computational methods to analyze voting behavior, particularly focusing on **silence patterns (abstention patterns)** to detect:

- **Implicit Consensus**: Shared opinions within a group that are not explicitly expressed
- **Hidden Minority Influence**: The hidden impact of minority groups on majority decisions
- **Group Pressure Dynamics**: Silent behaviors caused by group pressure

This project uses **Monte Carlo simulations** and **graph neural network modeling** to infer group dynamics characteristics from abstention data.

## Core Features

- **Multi-rule Voting System**: Supports various voting rules including Approval, Plurality, and weighted voting
- **Social Influence Graph**: Graph neural network model for establishing influence relationships between individuals
- **Pattern Detector**: Automatically detects patterns such as consensus, minority pressure, and oppressive dynamics
- **Large-scale Simulation Engine**: Supports parallel simulation of hundreds to thousands of agents
- **Visualization Analysis**: Provides various visualization tools including time series, distributions, and network graphs

## Installation

```bash
cd silence-decoder

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- numpy >= 1.24.0
- networkx >= 3.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0

## Usage

### Command Line Interface

```bash
# Run single simulation
python -m silence_decoder.cli run-sim \
    --num-agents 100 \
    --num-candidates 3 \
    --num-rounds 50 \
    --influence-density 0.3 \
    --output results/

# Run full experiment (consensus, minority, oppression)
python -m silence_decoder.cli run-experiment \
    --experiment-type all \
    --num-iterations 100 \
    --output results/

# Analyze saved data
python -m silence_decoder.cli analyze \
    --input results/voting_data.csv \
    --output reports/
```

### Python API

```python
from silence_decoder.src.agent import Agent
from silence_decoder.src.influence import InfluenceGraph
from silence_decoder.src.voting import VotingSystem
from silence_decoder.src.pattern_detector import PatternDetector
from silence_decoder.src.simulation import SimulationEngine

# Create agent
agent = Agent(
    agent_id=1,
    belief=[0.3, 0.7],
    influence_tolerance=0.5,
    opinion_strength=0.8
)

# Build influence graph
graph = InfluenceGraph()
graph.add_edge(1, 2, weight=0.7)
graph.add_edge(2, 3, weight=0.5)

# Run voting
voting_system = VotingSystem()
result = voting_system.run_vote(
    agents=[agent1, agent2, agent3],
    candidates=['A', 'B', 'C'],
    rule='approval'
)

# Detect patterns
detector = PatternDetector()
consensus = detector.detect_consensus(voting_data)
minority_pressure = detector.detect_minority_pressure(voting_data)
```

## Project Structure

```
silence-decoder/
├── src/                    # Core code
│   ├── agent.py           # Agent data model
│   ├── influence.py       # Influence graph
│   ├── voting.py          # Voting system
│   ├── simulation.py      # Simulation engine
│   ├── pattern_detector.py # Pattern detector
│   ├── analysis.py        # Analysis tools
│   ├── visualizer.py      # Visualization
│   ├── data_generator.py  # Data generator
│   └── cache.py           # Cache layer
├── experiments/            # Experiment scripts
│   ├── consensus_test.py
│   ├── minority_test.py
│   ├── oppression_test.py
│   └── config.json
├── notebooks/              # Jupyter notebooks
│   ├── 01-introduction.ipynb
│   ├── 02-consensus-detection.ipynb
│   └── 03-influence-analysis.ipynb
├── benchmarks/             # Performance tests
│   └── performance.py
├── tests/                  # Unit tests
│   ├── test_agent.py
│   ├── test_influence.py
│   ├── test_voting.py
│   └── test_pattern_detector.py
├── data/                   # Data files
├── models/                 # Trained models
├── docs/                   # Documentation
├── results/                # Experiment results
└── reports/                # Generated reports
```

## Experiment Design

### Consensus Detection Experiment

Verify whether the system can accurately detect and quantify consensus strength in scenarios with implicit consensus.

- **Scenario**: 100 agents, 3 candidates, known consensus strength
- **Metrics**: Detection accuracy, consensus strength estimation error

### Minority Influence Experiment

Study the ability of minority groups to change majority voting behavior.

- **Scenario**: 80 majority agents, 20 minority agents (with different beliefs)
- **Control Variables**: Minority cohesion, confidence, consistency
- **Output**: Minority influence threshold

### Oppressive Dynamics Experiment

Simulate oppressive group environments leading to suppressed voting behavior.

- **Scenario**: Dominant agent with high influence
- **Control Variables**: Oppression level (high/medium/low)
- **Output**: Correlation between oppression factor and observed abstention rate

## Deep Dive: Abstention Pattern Analysis

Abstention (silence) is not just "not voting" but an **information signal**:

| Abstention Type | Cause | Implicit Information |
|-----------------|-------|---------------------|
| Information-deficient Abstention | Lack of sufficient information to decide | Insufficient group information dissemination |
| Oppressive Abstention | Fear of expressing true opinions | Group oppression intensity |
| Consensus Abstention | Belief that others have made the correct decision | Strong implicit consensus |
| Protest Abstention | Dissatisfaction with candidates | Underlying discontent |

## Contributing

Contributions in the form of Issues and Pull Requests are welcome!

## License

MIT License - See LICENSE file for details.

---

## Support the Author

If you find this project helpful, feel free to buy me a coffee! ☕

![Buy Me a Coffee](buymeacoffee.png)

**Buy me a coffee (crypto)**

| Chain | Address |
|-------|---------|
| BTC | `bc1qc0f5tv577z7yt59tw8sqaq3tey98xehy32frzd` |
| ETH / USDT | `0x3b7b6c47491e4778157f0756102f134d05070704` |
| SOL | `6Xuk373zc6x6XWcAAuqvbWW92zabJdCmN3CSwpsVM6sd` |
