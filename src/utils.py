"""
utils.py
========
Helper functions for the quantum capstone project.

This module contains utility functions used across multiple modules.
Keeping utilities separate follows the DRY principle (Don't Repeat Yourself).
"""

import math


def validate_solution_string(solution, expected_length=4):
    """
    Validate that a solution string is properly formatted.

    WHY THIS MATTERS:
    Quantum circuits are expensive to build. Catching errors early
    prevents wasted computation and confusing bugs later.

    Args:
        solution: The bitstring to validate (e.g., "0010")
        expected_length: Expected number of bits

    Raises:
        ValueError: If solution is invalid
    """
    if len(solution) != expected_length:
        raise ValueError(
            f"solution must be exactly {expected_length} bits, got: {len(solution)}"
        )

    valid_chars = {"0", "1"}
    for i, char in enumerate(solution):
        if char not in valid_chars:
            raise ValueError(
                f"solution must contain only '0' or '1', got: {solution} "
                f"(invalid char '{char}' at position {i})"
            )


def calculate_optimal_iterations(num_qubits):
    """
    Calculate the optimal number of Grover iterations.

    THE MATH:
    Grover's algorithm rotates the state vector toward the solution.
    Each iteration rotates by ~2/sqrt(N) radians where N = 2^n.

    The optimal stopping point is when we've rotated by ~pi/2 radians
    (90 degrees), which gives maximum amplitude for the solution.

    Formula: iterations = floor(pi/4 * sqrt(N))

    WHY NOT MORE ITERATIONS?
    Grover's algorithm is periodic! After the optimal point, continuing
    actually DECREASES the solution probability. It's like a pendulum
    swinging past the highest point.

    Args:
        num_qubits: Number of qubits in the circuit

    Returns:
        int: Optimal number of iterations
    """
    n_states = 2 ** num_qubits
    iterations = int(math.floor((math.pi / 4) * math.sqrt(n_states)))
    return iterations


def get_qiskit_state_index(bitstring):
    """
    Convert a bitstring to Qiskit's statevector index.

    CRITICAL CONCEPT - Qiskit Bit Ordering:
    Qiskit uses LSB (Least Significant Bit) ordering:
    - q0 is the RIGHTMOST bit in the state label
    - q(n-1) is the LEFTmost bit

    Example for 4 qubits:
        String "0010" means: q3=0, q2=0, q1=1, q0=0
        Statevector index: reverse("0010") = "0100" = 4

    This is the #1 source of confusion for beginners!

    Args:
        bitstring: Binary string like "0010"

    Returns:
        int: The index in Qiskit's statevector
    """
    reversed_bits = bitstring[::-1]
    return int(reversed_bits, 2)


def format_probability(probability):
    """Format a probability value for display."""
    return f"{probability:.4f} ({probability * 100:.1f}%)"


def print_statevector_probabilities(statevector_data, highlight_index=None):
    """
    Pretty-print statevector probabilities with optional highlighting.

    Args:
        statevector_data: Array of probability amplitudes
        highlight_index: Index to highlight as the solution
    """
    num_states = len(statevector_data)
    num_qubits = int(math.log2(num_states))

    print(f"
Statevector Probabilities ({num_qubits} qubits, {num_states} states):")
    print("-" * 50)

    for i in range(num_states):
        bits = format(i, f"0{num_qubits}b")
        prob = abs(statevector_data[i]) ** 2
        bar_length = int(prob * 40)
        bar = "█" * bar_length
        marker = " <-- SOLUTION" if i == highlight_index else ""
        print(f"|{bits}>: {prob:.4f} {bar}{marker}")

    print("-" * 50)


# Self-test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("UTILITIES SELF-TEST")
    print("=" * 60)

    print("
[1] Testing validation:")
    try:
        validate_solution_string("0010")
        print("  '0010' -> VALID")
    except ValueError as e:
        print(f"  ERROR: {e}")

    try:
        validate_solution_string("0120")
    except ValueError as e:
        print(f"  '0120' -> Correctly rejected: {e}")

    print("
[2] Optimal iterations:")
    for n in [2, 3, 4, 5, 6]:
        iters = calculate_optimal_iterations(n)
        print(f"  {n} qubits -> {iters} iterations")

    print("
[3] Qiskit bit ordering:")
    test_cases = ["0000", "0001", "0010", "0100", "1000", "1111"]
    for bits in test_cases:
        idx = get_qiskit_state_index(bits)
        print(f"  '{bits}' -> index {idx} (binary: {format(idx, '04b')})")

    print("
" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
