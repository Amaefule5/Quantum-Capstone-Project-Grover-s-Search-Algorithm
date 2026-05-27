"""
gates.py
========
Custom quantum gate constructions for Grover's algorithm.

This module provides reusable gate patterns that appear multiple times
in Grover's algorithm. Extracting them into functions makes the main
code cleaner and easier to understand.

KEY CONCEPT: Multi-Controlled-Z (MCZ)
--------------------------------------
Qiskit doesn't have a direct MCZ gate, but we can build one using:
    MCZ = H(target) * MCX(controls, target) * H(target)

Why this works:
    H * X * H = Z

    The Hadamard gate converts between X-basis and Z-basis.
    So wrapping MCX with Hadamards converts it to MCZ.

This is a fundamental technique in quantum circuit design!
"""

from qiskit import QuantumCircuit


def apply_hadamard_all(circuit, qubits=None):
    """
    Apply Hadamard gates to all specified qubits.

    WHY HADAMARD?
    The Hadamard gate creates uniform superposition:
        H|0> = (|0> + |1>) / sqrt(2)

    When applied to all qubits starting from |0...0>, it creates
    an equal superposition of ALL possible states:
        H^{⊗n}|0...0> = (1/sqrt(2^n)) * Σ|x>

    This is the "quantum parallelism" that makes Grover's algorithm work!

    Args:
        circuit: The QuantumCircuit to modify
        qubits: List of qubit indices (default: all qubits)
    """
    if qubits is None:
        qubits = range(circuit.num_qubits)

    for qubit in qubits:
        circuit.h(qubit)


def apply_pauli_x_all(circuit, qubits=None):
    """
    Apply Pauli-X (NOT) gates to all specified qubits.

    WHY PAULI-X?
    The X gate flips |0> <-> |1>:
        X|0> = |1>
        X|1> = |0>

    In Grover's algorithm, X gates are used to:
    1. Transform states for oracle marking
    2. Implement the diffusion operator

    Args:
        circuit: The QuantumCircuit to modify
        qubits: List of qubit indices (default: all qubits)
    """
    if qubits is None:
        qubits = range(circuit.num_qubits)

    for qubit in qubits:
        circuit.x(qubit)


def apply_multi_controlled_z(circuit, control_qubits, target_qubit):
    """
    Apply a multi-controlled-Z gate using the H-MCX-H trick.

    THE TRICK EXPLAINED:
    -------------------
    A multi-controlled-Z applies a phase flip (-1) ONLY when ALL
    control qubits are |1> AND the target is |1>.

    But Qiskit only has multi-controlled-X (MCX), not MCZ.

    Solution: Use the identity H * X * H = Z

    Circuit:
        target: --H--[MCX]--H--
                    |
        controls: --*----*----*--

    This works because:
        - H converts |0>/|1> to |+>/|-> basis
        - MCX flips in X-basis (which is phase flip in Z-basis)
        - H converts back to Z-basis

    Args:
        circuit: The QuantumCircuit to modify
        control_qubits: List of control qubit indices
        target_qubit: Target qubit index

    Raises:
        ValueError: If target is in control list
    """
    if target_qubit in control_qubits:
        raise ValueError(
            f"Target qubit {target_qubit} cannot be in controls {control_qubits}"
        )

    # Step 1: Convert target to X-basis
    circuit.h(target_qubit)

    # Step 2: Apply multi-controlled-X
    circuit.mcx(control_qubits, target_qubit)

    # Step 3: Convert target back to Z-basis
    circuit.h(target_qubit)


def apply_oracle_marker(circuit, solution_string, target_qubit=None):
    """
    Apply the oracle that marks a specific solution with a phase flip.

    HOW IT WORKS:
    ------------
    The oracle needs to apply -1 phase to ONE specific state.

    Strategy:
    1. Flip qubits that are '0' in the solution using X gates
       (This transforms the solution state to |11...1>)
    2. Apply multi-controlled-Z to tag |11...1>
    3. Unflip the qubits (restore original state, now with phase)

    Example for solution "0010" (q3=0, q2=0, q1=1, q0=0):
        Step 1: X on q0, q2, q3 (flip the 0s)
                |0010> becomes |1111>
        Step 2: MCZ tags |1111> with -1 phase
        Step 3: X on q0, q2, q3 again (unflip)
                |1111> becomes |0010>, but with -1 phase!

    Args:
        circuit: The QuantumCircuit to modify
        solution_string: 4-bit string like "0010"
        target_qubit: Which qubit to use as MCZ target 
                      (default: last qubit, num_qubits-1)
    """
    num_qubits = circuit.num_qubits

    if target_qubit is None:
        target_qubit = num_qubits - 1

    # STEP 1: Flip qubits that are '0' in solution
    for i in range(num_qubits):
        if solution_string[i] == "0":
            circuit.x(i)

    # STEP 2: Apply multi-controlled-Z
    controls = list(range(num_qubits - 1))
    apply_multi_controlled_z(circuit, controls, target_qubit)

    # STEP 3: Unflip (undo step 1)
    for i in range(num_qubits):
        if solution_string[i] == "0":
            circuit.x(i)


def apply_diffusion_operator(circuit):
    """
    Apply the Grover diffusion (amplification) operator.

    WHAT IS DIFFUSION?
    -----------------
    The diffusion operator reflects all amplitudes about the average.

    Intuition:
    - After the oracle, the solution has negative amplitude
    - The average amplitude is slightly positive (most states are positive)
    - Reflecting about this average:
      * The negative solution becomes MORE positive (amplified!)
      * The positive non-solutions become LESS positive (attenuated)

    Mathematical form:
        D = 2|s><s| - I
    where |s> is the uniform superposition state.

    Implementation (same pattern as oracle):
        1. H on all (convert to X-basis)
        2. X on all (flip to |11...1>)
        3. MCZ on |11...1> (phase flip)
        4. X on all (unflip)
        5. H on all (convert back)

    Args:
        circuit: The QuantumCircuit to modify
    """
    num_qubits = circuit.num_qubits

    # Step 1: Hadamard on all qubits
    apply_hadamard_all(circuit)

    # Step 2: X on all qubits
    apply_pauli_x_all(circuit)

    # Step 3: Multi-controlled-Z
    controls = list(range(num_qubits - 1))
    target = num_qubits - 1
    apply_multi_controlled_z(circuit, controls, target)

    # Step 4: X on all (undo step 2)
    apply_pauli_x_all(circuit)

    # Step 5: Hadamard on all (undo step 1)
    apply_hadamard_all(circuit)


# Self-test
if __name__ == "__main__":
    print("=" * 60)
    print("GATES MODULE SELF-TEST")
    print("=" * 60)

    # Test 1: Hadamard all
    print("
[1] Testing apply_hadamard_all:")
    qc = QuantumCircuit(3)
    apply_hadamard_all(qc)
    print(qc.draw(output='text'))

    # Test 2: Pauli-X all
    print("
[2] Testing apply_pauli_x_all:")
    qc = QuantumCircuit(3)
    apply_pauli_x_all(qc)
    print(qc.draw(output='text'))

    # Test 3: Multi-controlled-Z
    print("
[3] Testing apply_multi_controlled_z:")
    qc = QuantumCircuit(4)
    apply_multi_controlled_z(qc, [0, 1, 2], 3)
    print(qc.draw(output='text'))

    # Test 4: Oracle marker
    print("
[4] Testing apply_oracle_marker (solution '0010'):")
    qc = QuantumCircuit(4)
    apply_oracle_marker(qc, "0010")
    print(qc.draw(output='text'))

    # Test 5: Diffusion
    print("
[5] Testing apply_diffusion_operator:")
    qc = QuantumCircuit(4)
    apply_diffusion_operator(qc)
    print(qc.draw(output='text'))

    print("
" + "=" * 60)
    print("All gate tests passed!")
    print("=" * 60)
