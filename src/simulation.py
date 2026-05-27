"""
================================================================================
SIMULATION.PY - Quantum Circuit Simulation & Visualization
================================================================================

This module provides tools to simulate quantum circuits and visualize results.
I separated these from the circuit construction because:
1. Simulation is about RUNNING circuits, not BUILDING them
2. Different users might want different visualization styles
3. It keeps the core algorithm code clean and focused

TWO TYPES OF SIMULATION:
1. Statevector simulation — gives exact probabilities (deterministic)
2. Aer (QASM) simulation — gives measurement statistics (probabilistic)

Statevector is better for understanding the algorithm mathematically.
Aer is better for understanding what real quantum measurements look like.

Author: [Your Name]
Course: Quantum Computing Capstone
Date: 2026
================================================================================
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def simulate_statevector(circuit: QuantumCircuit) -> dict:
    """
    Simulate a circuit using statevector method (exact probabilities).

    THOUGHT PROCESS — Why statevector simulation is useful:

    Statevector simulation computes the final quantum state exactly, without
    any randomness. This is incredibly valuable for:
    1. Debugging — you can see the exact amplitude of each basis state
    2. Verification — you can confirm the algorithm works with 100% certainty
    3. Understanding — you can trace how amplitudes evolve through the circuit

    MATHEMATICAL BASIS:
    A quantum circuit is a sequence of unitary operations U_1, U_2, ..., U_n.
    The final state is: |psi_final> = U_n * ... * U_2 * U_1 |0...0>

    Statevector simulation computes this matrix product explicitly.
    The probability of measuring state |x> is |<x|psi_final>|^2.

    LIMITATION:
    Statevector simulation requires storing 2^n complex numbers. For n=4,
    this is 16 numbers — trivial. For n=30, it's ~1 billion numbers — impossible
    on classical computers. This is why we need actual quantum computers!

    Args:
        circuit: The QuantumCircuit to simulate

    Returns:
        dict: Mapping from state labels (e.g., "0010") to probabilities
    """
    # Compute the statevector by applying all gates to |0...0>
    state = Statevector.from_instruction(circuit)

    # Extract probabilities (squared magnitudes of amplitudes)
    probabilities = np.abs(state.data) ** 2

    # Format as a dictionary with binary state labels
    num_qubits = circuit.num_qubits
    result = {}
    for i, prob in enumerate(probabilities):
        # Convert index to binary string, padded to num_qubits digits
        state_label = format(i, f"0{num_qubits}b")
        result[state_label] = float(prob)

    return result


def simulate_aer(circuit: QuantumCircuit, shots: int = 1024) -> dict:
    """
    Simulate a circuit using Qiskit Aer (measurement statistics).

    THOUGHT PROCESS — Why Aer simulation is different:

    Unlike statevector simulation which gives exact probabilities, Aer
    simulates what would happen if you ran the circuit on a real quantum
    device and measured it multiple times.

    Each "shot" is one complete run of the circuit from |0...0> to
    measurement. The result is a classical bitstring. After many shots,
    we count how often each bitstring appeared.

    This is PROBABILISTIC — running the same simulation twice gives
    slightly different results due to randomness. This mirrors real
    quantum hardware behavior.

    WHY USE AER:
    1. More realistic — matches what real quantum computers output
    2. Can model noise and errors (with noise models)
    3. Good for testing algorithms before running on expensive hardware

    Args:
        circuit: The QuantumCircuit to simulate (will have measurement added)
        shots: Number of times to run the circuit (default 1024)

    Returns:
        dict: Mapping from measured states to counts

    Raises:
        ImportError: If qiskit-aer is not installed
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        raise ImportError(
            "qiskit-aer is required for Aer simulation. "
            "Install it with: pip install qiskit-aer"
        )

    # Add measurement gates to all qubits
    # We create a COPY so the original circuit isn't modified
    measured_circuit = circuit.measure_all(inplace=False)

    # Create simulator and run
    simulator = AerSimulator()
    job = simulator.run(measured_circuit, shots=shots)
    result = job.result()

    # Get measurement counts
    counts = result.get_counts()

    return counts


def print_probabilities(probabilities: dict, solution: str = None, threshold: float = 0.01) -> None:
    """
    Pretty-print probability distribution from statevector simulation.

    I created this because raw numbers are hard to interpret. A visual
    bar chart in text form makes it immediately obvious which states
    have high probability.

    Args:
        probabilities: Dictionary of state -> probability
        solution: The expected solution state (highlighted if found)
        threshold: Only show states with probability above this
    """
    print(f"\n{'='*60}")
    print("STATEVECTOR PROBABILITIES (Exact)")
    print(f"{'='*60}")

    # Sort by probability (highest first)
    sorted_states = sorted(probabilities.items(), key=lambda x: -x[1])

    max_prob = max(probabilities.values())

    for state, prob in sorted_states:
        if prob < threshold:
            continue

        # Create visual bar
        bar_length = int(prob * 50)  # Scale to 50 characters max
        bar = "█" * bar_length

        # Mark solution
        marker = " ← SOLUTION!" if state == solution else ""

        # Mark highest probability
        if prob == max_prob and state != solution:
            marker = " ← HIGHEST"

        print(f"|{state}>: {prob:.4f} ({prob*100:5.1f}%) {bar}{marker}")

    print(f"{'='*60}\n")


def print_counts(counts: dict, shots: int, solution: str = None) -> None:
    """
    Pretty-print measurement counts from Aer simulation.

    Args:
        counts: Dictionary of state -> count
        shots: Total number of shots
        solution: The expected solution state (highlighted if found)
    """
    print(f"\n{'='*60}")
    print(f"MEASUREMENT COUNTS ({shots} shots)")
    print(f"{'='*60}")

    # Sort by count (highest first)
    sorted_states = sorted(counts.items(), key=lambda x: -x[1])

    max_count = max(counts.values())

    for state, count in sorted_states:
        percentage = (count / shots) * 100

        # Create visual bar
        bar_length = int(percentage / 2)  # Scale to 50 characters max
        bar = "█" * bar_length

        # Mark solution
        marker = " ← SOLUTION!" if state == solution else ""

        # Mark highest count
        if count == max_count and state != solution:
            marker = " ← MOST FREQUENT"

        print(f"|{state}>: {count:4d} ({percentage:5.1f}%) {bar}{marker}")

    print(f"{'='*60}\n")


def verify_solution(probabilities: dict, expected_solution: str) -> dict:
    """
    Verify that the Grover circuit found the correct solution.

    THOUGHT PROCESS — What does "correct" mean?

    For Grover's algorithm, "correct" means:
    1. The solution state has the highest probability
    2. The solution probability is above some threshold (e.g., 90%)

    However, due to the discrete nature of iterations, we might not get
    exactly 100%. For 4 qubits with 3 iterations, we expect ~97%.

    Args:
        probabilities: Dictionary of state -> probability
        expected_solution: The state we expect to find

    Returns:
        dict: Verification results with pass/fail status
    """
    # Find the state with highest probability
    max_prob_state = max(probabilities, key=probabilities.get)
    max_probability = probabilities[max_prob_state]

    # Check if solution matches
    solution_correct = (max_prob_state == expected_solution)

    # Check if probability is high enough
    # For 4 qubits with 3 iterations, theoretical max is ~0.97
    probability_threshold = 0.93  # Slightly below theoretical max for safety
    probability_sufficient = (max_probability >= probability_threshold)

    results = {
        "expected_solution": expected_solution,
        "found_solution": max_prob_state,
        "solution_correct": solution_correct,
        "max_probability": max_probability,
        "probability_sufficient": probability_sufficient,
        "passed": solution_correct and probability_sufficient
    }

    return results


def print_verification(results: dict) -> None:
    """
    Pretty-print verification results.

    Args:
        results: Dictionary from verify_solution()
    """
    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Expected solution: |{results['expected_solution']}>")
    print(f"Found solution:    |{results['found_solution']}>")
    print(f"Max probability:   {results['max_probability']:.4f} ({results['max_probability']*100:.1f}%)")
    print(f"\nSolution correct: {'PASS' if results['solution_correct'] else 'FAIL'}")
    print(f"Probability >= 93%: {'PASS' if results['probability_sufficient'] else 'FAIL'}")
    print(f"\nOverall: {'ALL TESTS PASSED!' if results['passed'] else 'TESTS FAILED'}")
    print(f"{'='*60}\n")
