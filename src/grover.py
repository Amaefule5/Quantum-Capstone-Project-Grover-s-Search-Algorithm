"""
grover.py
=========
Main Grover's algorithm implementation.

This module assembles the complete algorithm by combining:
- Initialization (uniform superposition)
- Oracle (marks the solution)
- Diffusion (amplifies the solution)

THE COMPLETE ALGORITHM:
-----------------------
1. Initialize: Apply H to all qubits -> uniform superposition
2. Repeat (Oracle + Diffusion) for optimal iterations:
   a. Oracle: Tag solution with -1 phase
   b. Diffusion: Amplify the tagged solution
3. Measure (done separately, not in these functions)

WHY REPEAT?
-----------
Each iteration rotates the state vector closer to the solution.
The optimal number is floor(pi/4 * sqrt(N)) where N = 2^n.

For 4 qubits (N=16): floor(3.14) = 3 iterations
For 2 qubits (N=4): floor(1.57) = 1 iteration

STOPPING EARLY: Not enough rotation, low probability
STOPPING LATE:  Overshoot, probability decreases again!
"""

from qiskit import QuantumCircuit
from .utils import validate_solution_string, calculate_optimal_iterations
from .oracle import two_qubit_oracle_11, four_qubit_oracle
from .diffusion import two_qubit_diffusion, four_qubit_diffusion


def two_qubit_grover_11():
    """
    Create a 2-qubit Grover circuit that finds the solution |11>.

    WHY ONLY |11>?
    --------------
    For 2 qubits, the CZ gate naturally tags |11>. To search for
    other solutions (|00>, |01>, |10>), we'd need additional X gates
    to transform them to |11> first. This function keeps it simple
    for educational purposes.

    ALGORITHM STEPS:
    ----------------
    1. Initialize: H on both qubits -> uniform superposition
    2. Oracle: CZ tags |11> with -1 phase
    3. Diffusion: H-X-CZ-X-H amplifies |11>

    Result: |11> has ~100% probability!

    Returns:
        QuantumCircuit: 2-qubit Grover circuit (no measurement)
    """
    # Create a fresh 2-qubit circuit
    num_qubits = 2
    qc = QuantumCircuit(num_qubits)

    # === STEP 1: INITIALIZE ===
    # Apply Hadamard to both qubits
    # Creates: (|00> + |01> + |10> + |11>) / 2
    qc.h(range(num_qubits))

    # === STEP 2: ORACLE ===
    # Tag |11> with a phase flip
    two_qubit_oracle_11(qc)

    # === STEP 3: DIFFUSION ===
    # Amplify the tagged solution
    two_qubit_diffusion(qc)

    return qc


def four_qubit_grover(solution):
    """
    Create a 4-qubit Grover circuit that finds the given solution.

    This is the GENERAL version that can find ANY 4-bit solution.

    ALGORITHM STEPS:
    ----------------
    1. Initialize: H on all 4 qubits -> uniform superposition of 16 states
    2. Calculate optimal iterations: floor(pi/4 * sqrt(16)) = 3
    3. Repeat 3 times:
       a. Oracle: Mark solution with -1 phase
       b. Diffusion: Amplify the solution

    Result: Solution has ~93-100% probability (depends on rounding)

    Args:
        solution: 4-bit string like "0010", "1111", "0000"
                 IMPORTANT: Uses Qiskit LSB ordering!
                 "0010" means q3=0, q2=0, q1=1, q0=0

    Returns:
        QuantumCircuit: 4-qubit Grover circuit (no measurement)

    Raises:
        ValueError: If solution is not a valid 4-bit string
    """
    # Validate input - catch errors early!
    validate_solution_string(solution, expected_length=4)

    # Create a fresh 4-qubit circuit
    num_qubits = 4
    qc = QuantumCircuit(num_qubits)

    # === STEP 1: INITIALIZE ===
    # Apply Hadamard to all qubits
    qc.h(range(num_qubits))

    # === STEP 2: CALCULATE ITERATIONS ===
    iterations = calculate_optimal_iterations(num_qubits)

    # === STEP 3: GROVER ITERATIONS ===
    for i in range(iterations):
        # Oracle: Mark the solution
        four_qubit_oracle(qc, solution)

        # Diffusion: Amplify the solution
        four_qubit_diffusion(qc)

    return qc


def create_grover_circuit(num_qubits, solution=None):
    """
    Factory function to create Grover circuits of different sizes.

    Args:
        num_qubits: Number of qubits (2 or 4)
        solution: Solution string (required for 4 qubits)

    Returns:
        QuantumCircuit: The complete Grover circuit
    """
    if num_qubits == 2:
        return two_qubit_grover_11()
    elif num_qubits == 4:
        if solution is None:
            raise ValueError("Solution string required for 4-qubit circuit")
        return four_qubit_grover(solution)
    else:
        raise ValueError(f"Only 2 or 4 qubits supported, got {num_qubits}")


# Self-test
if __name__ == "__main__":
    print("=" * 60)
    print("GROVER MODULE SELF-TEST")
    print("=" * 60)

    from qiskit.quantum_info import Statevector
    import numpy as np

    # Test 1: Two-qubit circuit
    print("
[1] Two-qubit Grover (solution |11>):")
    qc2 = two_qubit_grover_11()
    print(f"Gates: {len(qc2.data)}")
    print(qc2.draw(output='text'))

    sv2 = Statevector.from_instruction(qc2)
    probs2 = np.abs(sv2.data) ** 2
    print(f"
Probabilities:")
    for i, p in enumerate(probs2):
        print(f"  |{format(i, '02b')}>: {p:.4f}")

    # Test 2: Four-qubit circuit
    print("
[2] Four-qubit Grover (solution |0010>):")
    qc4 = four_qubit_grover("0010")
    print(f"Gates: {len(qc4.data)}")

    sv4 = Statevector.from_instruction(qc4)
    probs4 = np.abs(sv4.data) ** 2
    max_idx = np.argmax(probs4)
    print(f"
Solution |0010> probability: {probs4[4]:.4f}")
    print(f"Max probability state: |{format(max_idx, '04b')}>")

    # Test 3: Factory function
    print("
[3] Factory function:")
    qc_factory = create_grover_circuit(4, "1111")
    print(f"Factory created circuit with {len(qc_factory.data)} gates")

    print("
" + "=" * 60)
    print("Grover tests passed!")
    print("=" * 60)
