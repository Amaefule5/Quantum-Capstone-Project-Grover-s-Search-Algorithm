"""
oracle.py
=========
Oracle implementations for Grover's algorithm.

WHAT IS AN ORACLE?
------------------
In quantum computing, an "oracle" is a black-box function that
identifies the solution to a problem. In Grover's algorithm:

    Oracle|x> = -|x>  if x is the solution
    Oracle|x> =  |x>  if x is NOT the solution

The oracle applies a PHASE FLIP (-1) to the solution state.
This doesn't change probabilities yet, but changes the PHASE.

WHY PHASE MATTERS:
-----------------
Phase alone doesn't affect measurement. But when combined with
the diffusion operator, the negative phase causes CONSTRUCTIVE
interference for the solution and DESTRUCTIVE interference for
non-solutions. This amplifies the solution's probability!
"""

from qiskit import QuantumCircuit
from .gates import apply_oracle_marker


def two_qubit_oracle_11(circuit):
    """
    Apply the oracle for 2-qubit Grover searching for |11>.

    WHY THIS IS SPECIAL:
    -------------------
    For 2 qubits searching for |11>, the Controlled-Z (CZ) gate
    IS the oracle! This is a happy coincidence that doesn't work
    for other solutions.

    How CZ tags |11>:
        CZ|00> = |00>     (no phase change)
        CZ|01> = |01>     (no phase change)
        CZ|10> = |10>     (no phase change)
        CZ|11> = -|11>    (PHASE FLIP! Solution tagged!)

    The CZ gate applies a Z gate to the target ONLY when the
    control is |1>. With both qubits as control and target, it
    applies phase flip ONLY to |11>.

    Args:
        circuit: 2-qubit QuantumCircuit to modify

    Raises:
        ValueError: If circuit doesn't have exactly 2 qubits
    """
    if circuit.num_qubits != 2:
        raise ValueError(
            f"Two-qubit oracle requires exactly 2 qubits, got {circuit.num_qubits}"
        )

    # Apply Controlled-Z between q0 and q1
    circuit.cz(0, 1)


def four_qubit_oracle(circuit, solution):
    """
    Apply the general oracle for 4-qubit Grover.

    This oracle can mark ANY 4-bit solution string.

    HOW IT WORKS:
    ------------
    1. We need to apply -1 phase to exactly ONE state out of 16
    2. We can't directly target arbitrary states
    3. Solution: Transform our target state to |1111>, tag it, then transform back

    The transformation uses X gates on qubits that are '0' in the solution:
        Solution "0010" -> qubits 0,2,3 are 0 -> flip them
        |0010> -> X on q0,q2,q3 -> |1111>

    Then apply multi-controlled-Z (which tags |1111>)
    Then unflip (X again on same qubits)

    Result: |0010> now has -1 phase, all others unchanged!

    Args:
        circuit: 4-qubit QuantumCircuit to modify
        solution: 4-bit string like "0010", "1111"

    Raises:
        ValueError: If circuit doesn't have 4 qubits or solution is invalid
    """
    if circuit.num_qubits != 4:
        raise ValueError(
            f"Four-qubit oracle requires exactly 4 qubits, got {circuit.num_qubits}"
        )

    # Use the general oracle marker from gates.py
    apply_oracle_marker(circuit, solution)


def create_oracle_circuit(num_qubits, solution=None):
    """
    Create a standalone oracle circuit for inspection/testing.

    Sometimes it's useful to see the oracle in isolation to
    understand what it's doing.

    Args:
        num_qubits: Number of qubits (2 or 4)
        solution: Solution string (required for 4 qubits)

    Returns:
        QuantumCircuit: The oracle circuit
    """
    qc = QuantumCircuit(num_qubits)

    if num_qubits == 2:
        two_qubit_oracle_11(qc)
    elif num_qubits == 4:
        if solution is None:
            raise ValueError("Solution string required for 4-qubit oracle")
        four_qubit_oracle(qc, solution)
    else:
        raise ValueError(f"Only 2 or 4 qubits supported, got {num_qubits}")

    return qc


# Self-test
if __name__ == "__main__":
    print("=" * 60)
    print("ORACLE MODULE SELF-TEST")
    print("=" * 60)

    from qiskit.quantum_info import Statevector
    import numpy as np

    # Test 1: Two-qubit oracle
    print("\n   [1] Two-qubit oracle (solution |11>):")
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    two_qubit_oracle_11(qc)
    sv = Statevector.from_instruction(qc)
    print(f"Statevector: {sv.data}")
    print(f"|11> amplitude: {sv.data[3]} (should be -1+0j)")

    # Test 2: Four-qubit oracle
    print("\n   [2] Four-qubit oracle (solution |0010>):")
    qc = QuantumCircuit(4)
    qc.x(1)  # q1 = 1
    four_qubit_oracle(qc, "0010")
    sv = Statevector.from_instruction(qc)
    target_index = 4
    print(f"State |0010> amplitude: {sv.data[target_index]}")
    print(f"Should have -1 phase: {np.isclose(sv.data[target_index], -1.0)}")

    # Test 3: Standalone oracle circuit
    print("\n   [3] Standalone oracle circuit:")
    oracle = create_oracle_circuit(4, "1100")
    print(oracle.draw(output='text'))

    print("\n" + "=" * 60)
    print("Oracle tests passed!")
    print("=" * 60)
