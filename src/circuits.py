"""
circuits.py
===========
Pre-built circuit templates and visualization helpers.

This module provides ready-to-use circuits and tools for:
- Visualizing circuits as text or images
- Creating measurement circuits
- Running simulations and displaying results

WHY SEPARATE VISUALIZATION?
---------------------------
Keeping visualization separate from algorithm logic follows the
Single Responsibility Principle. The grover.py module handles the
quantum algorithm; this module handles how we see and test it.
"""

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def add_measurement(circuit, inplace=False):
    """
    Add measurement gates to a circuit.

    WHY MEASUREMENT IS SEPARATE:
    ---------------------------
    In the capstone project, circuits are built WITHOUT measurement.
    This allows us to:
    1. Inspect the statevector (exact amplitudes)
    2. Add measurement only when we want to simulate "real" quantum execution

    Measurement collapses the quantum state, so we add it as the
    final step before running on a simulator.

    Args:
        circuit: The QuantumCircuit to measure
        inplace: If True, modify original; if False, return copy

    Returns:
        QuantumCircuit: Circuit with measurement added
    """
    if inplace:
        circuit.measure_all()
        return circuit
    else:
        measured = circuit.copy()
        measured.measure_all()
        return measured


def draw_circuit_text(circuit, title=None):
    """
    Generate a text representation of a circuit.

    Text diagrams are great for:
    - Quick debugging in terminal/VS Code
    - Copy-pasting into documentation
    - Understanding gate sequences without images

    Args:
        circuit: The QuantumCircuit to draw
        title: Optional title to print above diagram

    Returns:
        str: Text representation of the circuit
    """
    if title:
        print(f"{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")

    diagram = circuit.draw(output='text')
    print(diagram)

    print(f"\n  Qubits: {circuit.num_qubits}")
    print(f"  Gates: {len(circuit.data)}")

    return str(diagram)


def print_circuit_info(circuit):
    """
    Print detailed information about a circuit.

    Args:
        circuit: The QuantumCircuit to analyze
    """
    print(f"{'='*50}")
    print("  CIRCUIT INFORMATION")
    print(f"{'='*50}")

    print(f"  Number of qubits: {circuit.num_qubits}")
    print(f"  Number of classical bits: {circuit.num_clbits}")
    print(f"  Total operations: {len(circuit.data)}")
    print(f"  Circuit depth: {circuit.depth()}")

    # Count gate types
    gate_counts = {}
    for instr in circuit.data:
        name = instr.name
        gate_counts[name] = gate_counts.get(name, 0) + 1

    print(f"\n  Gate breakdown:")
    for gate, count in sorted(gate_counts.items()):
        print(f"    {gate}: {count}")

    print(f"{'='*50}")


def create_comparison_circuits():
    """
    Create circuits for comparing different stages of Grover's algorithm.

    Returns:
        dict: Dictionary of named circuits
    """
    from .grover import two_qubit_grover_11
    from .oracle import two_qubit_oracle_11
    from .diffusion import two_qubit_diffusion

    circuits = {}

    # 1. Initialization only
    qc_init = QuantumCircuit(2)
    qc_init.h(range(2))
    circuits["initialization"] = qc_init

    # 2. After oracle
    qc_oracle = QuantumCircuit(2)
    qc_oracle.h(range(2))
    two_qubit_oracle_11(qc_oracle)
    circuits["after_oracle"] = qc_oracle

    # 3. After diffusion
    qc_diffusion = QuantumCircuit(2)
    qc_diffusion.h(range(2))
    two_qubit_oracle_11(qc_diffusion)
    two_qubit_diffusion(qc_diffusion)
    circuits["after_diffusion"] = qc_diffusion

    # 4. Complete
    circuits["complete"] = two_qubit_grover_11()

    return circuits


# Self-test
if __name__ == "__main__":
    print("=" * 60)
    print("CIRCUITS MODULE SELF-TEST")
    print("=" * 60)

    from .grover import two_qubit_grover_11

    # Test 1: Add measurement
    print("\n   [1] Adding measurement:")
    qc = two_qubit_grover_11()
    qc_meas = add_measurement(qc, inplace=False)
    print(f"Original gates: {len(qc.data)}")
    print(f"Measured gates: {len(qc_meas.data)}")

    # Test 2: Text drawing
    print("\n   [2] Text diagram:")
    draw_circuit_text(qc, "Two-Qubit Grover")

    # Test 3: Circuit info
    print("\n   [3] Circuit info:")
    print_circuit_info(qc)

    # Test 4: Comparison circuits
    print("\n   [4] Comparison circuits:")
    comparisons = create_comparison_circuits()
    for name, circ in comparisons.items():
        print(f"  {name}: {len(circ.data)} gates")

    print("\n   " + "=" * 60)
    print("Circuits tests passed!")
    print("=" * 60)
