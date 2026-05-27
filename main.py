"""
main.py
=======
Main entry point for the Quantum Capstone Project.

This script demonstrates the complete Grover's algorithm implementation
and serves as a reference for how to use the various modules.

RUNNING THIS SCRIPT:
-------------------
    python main.py

WHAT IT DOES:
------------
1. Builds 2-qubit and 4-qubit Grover circuits
2. Displays circuit diagrams
3. Runs statevector simulation (exact probabilities)
4. Runs measurement simulation (shot-based statistics)
5. Prints formatted results

MODULE ORGANIZATION:
-------------------
The project is split into focused modules:
- src/utils.py      : Validation and calculation helpers
- src/gates.py      : Reusable quantum gate patterns
- src/oracle.py     : Oracle implementations (mark solutions)
- src/diffusion.py  : Diffusion operators (amplify solutions)
- src/grover.py     : Main algorithm assembly
- src/circuits.py   : Visualization and measurement tools
- tests/            : Comprehensive test suites
"""

import sys
import math

# Ensure src/ is in path (for running without package install)
sys.path.insert(0, 'src')

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

# Import our modules
from src.grover import two_qubit_grover_11, four_qubit_grover
from src.circuits import (
    add_measurement,
    draw_circuit_text,
    print_circuit_info
)
from src.utils import print_statevector_probabilities


def run_two_qubit_demo():
    """
    Demonstrate the 2-qubit Grover algorithm.

    The 2-qubit case is special:
    - Only 1 iteration needed
    - Achieves exactly 100% probability
    - Uses simple CZ gate as oracle
    """
    print("=" * 60)
    print("  2-QUBIT GROVER'S ALGORITHM")
    print("  Solution: |11>")
    print("=" * 60)

    # Build the circuit
    print("\n[STEP 1] Building circuit...")
    qc = two_qubit_grover_11()

    # Show circuit info
    print_circuit_info(qc)

    # Draw the circuit
    draw_circuit_text(qc, "2-Qubit Grover Circuit")

    # Run statevector simulation
    print("\n[STEP 2] Statevector simulation (exact probabilities):")
    state = Statevector.from_instruction(qc)
    print_statevector_probabilities(state.data, highlight_index=3)

    # Verify 100% success
    probs = abs(state.data) ** 2
    print(f"\n Solution |11> probability: {probs[3]*100:.1f}%")
    print(f"  Expected: 100%")
    print(f"  Status: {'PASS' if probs[3] > 0.999 else 'FAIL'}")


def run_four_qubit_demo(solution="0010"):
    """
    Demonstrate the 4-qubit Grover algorithm.

    The 4-qubit case is the general solution:
    - 3 iterations (optimal)
    - Achieves ~93-97% probability
    - Uses multi-controlled operations

    Args:
        solution: 4-bit string to search for (default "0010")
    """
    print("\n" + "=" * 60)
    print(f"  4-QUBIT GROVER'S ALGORITHM")
    print(f"  Solution: |{solution}>")
    print("=" * 60)

    # Build the circuit
    print(f"\n[STEP 1] Building circuit for |{solution}>...")
    qc = four_qubit_grover(solution)

    # Show circuit info
    print_circuit_info(qc)

    # Calculate expected state index (Qiskit LSB ordering)
    expected_index = int(solution[::-1], 2)
    print(f"  Expected statevector index: {expected_index}")
    print(f"  Optimal iterations: {int(math.floor((math.pi/4) * math.sqrt(16)))}")

    # Draw the circuit (truncated for display)
    print("\n[STEP 2] Circuit diagram (first 20 gates shown):")
    print("  [Circuit has too many gates for terminal display]")
    print("  Use: qc.draw(output='mpl', filename='circuit.png') for image")

    # Run statevector simulation
    print("\n[STEP 3] Statevector simulation (exact probabilities):")
    state = Statevector.from_instruction(qc)
    print_statevector_probabilities(state.data, highlight_index=expected_index)

    # Check solution probability
    probs = abs(state.data) ** 2
    solution_prob = probs[expected_index]
    max_idx = probs.argmax()

    print(f"\n Solution |{solution}> probability: {solution_prob*100:.1f}%")
    print(f"  Maximum probability state: |{format(max_idx, '04b')}>")
    print(f"  Expected: |{solution}> (index {expected_index})")
    print(f"  Status: {'PASS' if max_idx == expected_index and solution_prob > 0.93 else 'FAIL'}")

    # Run measurement simulation (if qiskit-aer available)
    print("\n[STEP 4] Measurement simulation (1024 shots):")
    try:
        from qiskit_aer import AerSimulator

        # Add measurement
        qc_meas = add_measurement(qc, inplace=False)

        # Run simulator
        sim = AerSimulator()
        result = sim.run(qc_meas, shots=1024).result()
        counts = result.get_counts()

        # Display top results
        print("\n   Top measurement results:")
        for state_label, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            pct = (count / 1024) * 100
            marker = " <- SOLUTION!" if state_label == solution[::-1] else ""
            print(f"    |{state_label}>: {count:4d} shots ({pct:5.1f}%){marker}")

        # Check if solution was most frequent
        most_frequent = max(counts, key=counts.get)
        print(f"\n  Most frequent result: |{most_frequent}>")
        print(f"\n  Status: {'PASS' if most_frequent == solution[::-1] else 'FAIL'}")

    except ImportError:
        print("  [qiskit-aer not installed, skipping measurement simulation]")
        print("  Install with: pip install qiskit-aer")


def run_all_tests_summary():
    """Run a quick summary of all test cases."""
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)

    test_cases = [
        ("0000", "All zeros"),
        ("1111", "All ones"),
        ("0010", "Single one (middle)"),
        ("1000", "Single one (first bit)"),
        ("0101", "Alternating pattern"),
        ("1010", "Alternating pattern (reversed)"),
    ]

    print("\n   Testing multiple solutions:")
    print(f"  {'Solution':<10} {'Index':<8} {'Probability':<15} {'Status'}")
    print("  " + "-" * 50)

    for solution, description in test_cases:
        qc = four_qubit_grover(solution)
        state = Statevector.from_instruction(qc)
        probs = abs(state.data) ** 2

        expected_idx = int(solution[::-1], 2)
        solution_prob = probs[expected_idx]
        max_idx = probs.argmax()

        status = "PASS" if max_idx == expected_idx and solution_prob > 0.93 else "FAIL"

        print(f"  |{solution}>   {expected_idx:<8} {solution_prob*100:6.1f}%        {status}")

    print("\n   All solutions tested!")


def main():
    """
    Main entry point.

    Runs the complete demonstration of Grover's algorithm.
    """
    print("\n")
    print("*" * 60)
    print("*  QUANTUM CAPSTONE PROJECT")
    print("*  Grover's Search Algorithm Implementation")
    print("*  Using Qiskit 2.0")
    print("*" * 60)

    # Run 2-qubit demo
    run_two_qubit_demo()

    # Run 4-qubit demo
    run_four_qubit_demo("0010")

    # Run test summary
    run_all_tests_summary()

    print("\n" + "=" * 60)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\n   To run full test suite:")
    print("    python -m unittest discover tests/ -v")
    print("\n   To run specific test:")
    print("    python -m unittest tests.test_four_qubit -v")
    print("\n   To visualize a circuit:")
    print('    python -c "from src.grover import four_qubit_grover; qc = four_qubit_grover(\'0010\'); qc.draw(output=\'mpl\')"')
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
