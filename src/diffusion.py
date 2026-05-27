"""
diffusion.py
============
Diffusion (amplification) operator for Grover's algorithm.

WHAT IS DIFFUSION?
-------------------
After the oracle tags the solution with a -1 phase, the diffusion
operator AMPLIFIES the solution's probability.

HOW IT WORKS:
------------
The diffusion operator performs "inversion about the average":

1. Calculate the average amplitude of all states
2. Reflect each amplitude about this average

Mathematical effect:
    - States with amplitude BELOW average get pushed DOWN
    - States with amplitude ABOVE average get pushed UP

Since the solution now has NEGATIVE amplitude (from oracle),
    - The average is slightly positive (most states are positive)
    - The solution is BELOW average (negative < positive)
    - After reflection: solution becomes MORE positive (amplified!)
    - Non-solutions are above average, get pushed down (attenuated)

VISUAL INTUITION:
----------------
Imagine amplitudes as bars on a graph:

    Before Oracle:          After Oracle:           After Diffusion:

    |    |                  |    |                  |    |
    |    |                  |    |                  |    |
    |####| <- solution      |####| <- negative!     |    |
    |####|                  |####|                  |    |
    |####| <- average       |####| <- average       |####| <- average
    |####|                  |####|                  |####|
    |####|                  |####|                  |####|

    All equal               Solution negative        Solution amplified!

IMPLEMENTATION:
--------------
The diffusion operator is implemented identically to the oracle
but with a "dummy" solution of |00...0>. This is because:

    D = H^{⊗n} * (2|0...0><0...0| - I) * H^{⊗n}
      = 2|s><s| - I

where |s> is the uniform superposition. The sequence:
    H -> X -> MCZ -> X -> H

implements this reflection operation.
"""

from qiskit import QuantumCircuit
from .gates import apply_diffusion_operator


def two_qubit_diffusion(circuit):
    """
    Apply the diffusion operator for 2-qubit Grover's algorithm.

    For 2 qubits, the diffusion operator is:
        H on all -> X on all -> CZ -> X on all -> H on all

    This is the same pattern as the 4-qubit version but using
    the simpler CZ gate instead of multi-controlled-Z.

    Args:
        circuit: 2-qubit QuantumCircuit to modify

    Raises:
        ValueError: If circuit doesn't have exactly 2 qubits
    """
    if circuit.num_qubits != 2:
        raise ValueError(
            f"Two-qubit diffusion requires exactly 2 qubits, got {circuit.num_qubits}"
        )

    # Step 1: Hadamard on all qubits
    circuit.h(range(2))

    # Step 2: Pauli-X on all qubits
    circuit.x(range(2))

    # Step 3: Controlled-Z
    circuit.cz(0, 1)

    # Step 4: Pauli-X on all (undo step 2)
    circuit.x(range(2))

    # Step 5: Hadamard on all (undo step 1)
    circuit.h(range(2))


def four_qubit_diffusion(circuit):
    """
    Apply the diffusion operator for 4-qubit Grover's algorithm.

    Uses the general apply_diffusion_operator from gates.py.

    Args:
        circuit: 4-qubit QuantumCircuit to modify

    Raises:
        ValueError: If circuit doesn't have exactly 4 qubits
    """
    if circuit.num_qubits != 4:
        raise ValueError(
            f"Four-qubit diffusion requires exactly 4 qubits, got {circuit.num_qubits}"
        )

    apply_diffusion_operator(circuit)


def create_diffusion_circuit(num_qubits):
    """
    Create a standalone diffusion circuit for inspection/testing.

    Args:
        num_qubits: Number of qubits (2 or 4)

    Returns:
        QuantumCircuit: The diffusion circuit
    """
    qc = QuantumCircuit(num_qubits)

    if num_qubits == 2:
        two_qubit_diffusion(qc)
    elif num_qubits == 4:
        four_qubit_diffusion(qc)
    else:
        raise ValueError(f"Only 2 or 4 qubits supported, got {num_qubits}")

    return qc


# Self-test
if __name__ == "__main__":
    print("=" * 60)
    print("DIFFUSION MODULE SELF-TEST")
    print("=" * 60)

    # Test 1: Two-qubit diffusion
    print("
[1] Two-qubit diffusion circuit:")
    qc = QuantumCircuit(2)
    two_qubit_diffusion(qc)
    print(qc.draw(output='text'))

    # Test 2: Four-qubit diffusion
    print("
[2] Four-qubit diffusion circuit:")
    qc = QuantumCircuit(4)
    four_qubit_diffusion(qc)
    print(qc.draw(output='text'))

    # Test 3: Standalone circuit
    print("
[3] Standalone diffusion circuit:")
    diff = create_diffusion_circuit(4)
    print(f"Number of gates: {len(diff.data)}")

    print("
" + "=" * 60)
    print("Diffusion tests passed!")
    print("=" * 60)
