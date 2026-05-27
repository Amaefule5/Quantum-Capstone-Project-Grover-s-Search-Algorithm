# Quantum Capstone Project: Grover's Search Algorithm

A complete, modular implementation of Grover's quantum search algorithm using Qiskit 2.0.

## Project Structure

```
quantum-capstone/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── main.py                  # Entry point - runs full demonstration
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── utils.py             # Validation, calculations, helpers
│   ├── gates.py             # Quantum gate constructions
│   ├── oracle.py            # Oracle implementations
│   ├── diffusion.py         # Diffusion operators
│   ├── grover.py            # Main algorithm assembly
│   └── circuits.py          # Visualization and measurement
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_two_qubit.py    # 2-qubit algorithm tests
│   ├── test_four_qubit.py  # 4-qubit algorithm tests
│   └── test_gates.py       # Individual gate tests
└── notebooks/               # Jupyter notebooks
    └── 01_introduction.ipynb # Step-by-step walkthrough
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate (macOS/Linux)
source .venv/bin/activate

# 3. Activate (Windows)
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the main demonstration
python main.py

# 6. Run all tests
python -m unittest discover tests/ -v
```

## What is Grover's Algorithm?

Grover's algorithm provides a **quadratic speedup** for searching unsorted databases.

| Approach | 4 Qubits (N=16) |
|----------|----------------|
| Classical | Up to 16 checks |
| Grover's Quantum | ~3 iterations |

## Key Concepts

- **Superposition**: Qubits exist in multiple states simultaneously
- **Oracle**: Tags the solution with a phase flip (-1)
- **Diffusion**: Amplifies the tagged solution's probability
- **Interference**: Phase differences create constructive/destructive patterns

## License

CC BY-NC-ND 4.0 - QuLearnLabs 2026

Amaefule Capstone Project

AI-SEQ Course
