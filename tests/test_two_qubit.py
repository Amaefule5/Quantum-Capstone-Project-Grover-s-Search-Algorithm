"""
test_two_qubit.py
=================
Unit tests for the 2-qubit Grover's algorithm.

These tests verify that:
1. The circuit finds |11> with 100% probability
2. All other states have 0% probability
3. The circuit uses only allowed gates

TESTING PHILOSOPHY:
------------------
For quantum circuits, we test:
- Statevector (exact mathematical result)
- Gate composition (structural correctness)
- Probabilities (physical measurement outcomes)
"""

import unittest
import numpy as np
from qiskit.quantum_info import Statevector

from src.grover import two_qubit_grover_11


class TestTwoQubitGrover(unittest.TestCase):
    """
    Test suite for 2-qubit Grover's algorithm.

    The 2-qubit case is special because:
    - Only 1 iteration needed (optimal)
    - CZ gate directly serves as oracle
    - Achieves 100% success probability (theoretical maximum)
    """

    def test_solution_probability_is_100_percent(self):
        """
        Verify that |11> is found with exactly 100% probability.

        WHY 100%?
        ---------
        For 2 qubits with 1 iteration, the math works out such that
        1 iteration gives exactly 100%. This is a special case that
        doesn't generalize to more qubits.

        For 4 qubits, we only get ~93% because 3 iterations is an
        approximation of the optimal pi/4 * sqrt(16) = 3.14...
        """
        qc = two_qubit_grover_11()
        state = Statevector.from_instruction(qc)
        probabilities = np.abs(state.data) ** 2

        solution_index = 3
        solution_probability = probabilities[solution_index]

        self.assertAlmostEqual(
            solution_probability, 
            1.0, 
            places=10,
            msg=f"Expected |11> to have 100% probability, got {solution_probability}"
        )

    def test_other_states_have_zero_probability(self):
        """
        Verify that non-solution states have exactly 0% probability.
        """
        qc = two_qubit_grover_11()
        state = Statevector.from_instruction(qc)
        probabilities = np.abs(state.data) ** 2

        for i in range(4):
            if i != 3:
                self.assertAlmostEqual(
                    probabilities[i],
                    0.0,
                    places=10,
                    msg=f"State |{format(i, '02b')}> should have 0% probability"
                )

    def test_circuit_has_correct_number_of_gates(self):
        """
        Verify the gate count matches expected structure.

        Expected: 2 H + 1 CZ + 2 H + 2 X + 1 CZ + 2 X + 2 H = 12 gates
        """
        qc = two_qubit_grover_11()
        total_gates = len(qc.data)

        self.assertEqual(
            total_gates,
            12,
            f"Expected 12 gates, got {total_gates}"
        )

    def test_circuit_uses_only_allowed_gates(self):
        """Verify only H, X, and CZ gates are used."""
        qc = two_qubit_grover_11()
        allowed = {"h", "x", "cz"}

        for instruction in qc.data:
            gate_name = instruction.name
            self.assertIn(
                gate_name,
                allowed,
                f"Gate '{gate_name}' is not in allowed set {allowed}"
            )

    def test_circuit_has_two_qubits(self):
        """Verify the circuit operates on exactly 2 qubits."""
        qc = two_qubit_grover_11()
        self.assertEqual(qc.num_qubits, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
