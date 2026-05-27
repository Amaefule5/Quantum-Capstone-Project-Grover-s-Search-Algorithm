"""
test_four_qubit.py
==================
Unit tests for the 4-qubit Grover's algorithm.

These tests verify that:
1. The circuit finds the correct solution for ALL 16 possible inputs
2. The solution probability is at least 93% (optimal for 4 qubits)
3. Only allowed gates (h, x, mcx) are used
4. Bit ordering is correctly handled (Qiskit LSB convention)

TESTING STRATEGY:
----------------
We test ALL 16 possible solutions because:
- The oracle must work for any input
- Different inputs require different X gate patterns
- This ensures the flip->tag->unflip pattern is correct

BIT ORDERING TEST:
-----------------
The test `expected_values` dictionary encodes Qiskit's LSB ordering:
    "0001" -> reversed "1000" -> index 8
    "0010" -> reversed "0100" -> index 4
    "1000" -> reversed "0001" -> index 1

This is the most critical test because bit ordering is the #1
source of bugs in quantum circuit programming!
"""

import unittest
import numpy as np
from qiskit.quantum_info import Statevector

from src.grover import four_qubit_grover


class TestFourQubitGrover(unittest.TestCase):
    """
    Comprehensive test suite for 4-qubit Grover's algorithm.
    """

    def setUp(self):
        """
        Set up test fixtures before each test method.

        Creates the expected mapping between bitstrings and
        Qiskit statevector indices.

        QISKIT BIT ORDERING EXPLAINED:
        -----------------------------
        In Qiskit, the state |q3 q2 q1 q0> is labeled as a binary
        number where q0 is the LEAST significant bit (rightmost).

        Example:
            State |0010> means:
                q3 = 0 (leftmost, most significant)
                q2 = 0
                q1 = 1
                q0 = 0 (rightmost, least significant)

            The statevector index is computed by reversing the string!
            "0010" reversed = "0100" = 4

            So |0010> is at index 4!

        This is why we use [::-1] (string reverse) in the test.
        """
        self.expected_values = {
            format(i, "04b"): int(format(i, "04b")[::-1], 2)
            for i in range(16)
        }

    def test_all_solutions_have_high_probability(self):
        """
        Test that EVERY possible 4-bit solution is found with >= 93% probability.

        WHY 93%?
        ---------
        The optimal iteration count is floor(pi/4 * sqrt(16)) = 3.
        But the exact optimal is pi/4 * 4 = 3.14159...

        Since we can only do integer iterations:
        - 3 iterations gives ~96.5% probability
        - 4 iterations would overshoot to ~90%

        We use 3 iterations and accept ~93-97% probability.
        """
        for bitstring, expected_index in self.expected_values.items():
            with self.subTest(solution=bitstring):
                qc = four_qubit_grover(bitstring)
                state = Statevector.from_instruction(qc)
                probabilities = np.abs(state.data) ** 2

                max_prob = np.max(probabilities)
                solution_state = np.argmax(probabilities)

                self.assertEqual(
                    solution_state,
                    expected_index,
                    f"Solution {bitstring}: expected index {expected_index}, "
                    f"got {solution_state}"
                )

                self.assertGreaterEqual(
                    max_prob,
                    0.93,
                    f"Solution {bitstring}: probability {max_prob} < 0.93"
                )

    def test_specific_solution_0010(self):
        """
        Detailed test for solution "0010" with full output.
        """
        solution = "0010"
        expected_index = self.expected_values[solution]

        qc = four_qubit_grover(solution)
        state = Statevector.from_instruction(qc)
        probabilities = np.abs(state.data) ** 2

        print(f"
Probability distribution for solution |{solution}>:")
        print("-" * 40)
        for i in range(16):
            bits = format(i, '04b')
            marker = " <- SOLUTION" if i == expected_index else ""
            bar = "█" * int(probabilities[i] * 30)
            print(f"|{bits}>: {probabilities[i]:.4f} {bar}{marker}")
        print("-" * 40)

        max_idx = np.argmax(probabilities)
        self.assertEqual(max_idx, expected_index)
        self.assertGreaterEqual(probabilities[expected_index], 0.93)

    def test_circuit_uses_only_allowed_gates(self):
        """
        Verify that only H, X, and MCX gates are used.

        WHY THIS CONSTRAINT?
        -------------------
        The test enforces a minimal gate set to ensure:
        1. We're using the standard Grover construction
        2. The H-MCX-H pattern for multi-controlled-Z
        """
        allowed_gates = {"h", "x", "mcx"}

        for bitstring in self.expected_values:
            with self.subTest(solution=bitstring):
                qc = four_qubit_grover(bitstring)

                for instruction in qc.data:
                    gate_name = instruction.name
                    self.assertIn(
                        gate_name,
                        allowed_gates,
                        f"Solution {bitstring}: found disallowed gate '{gate_name}'"
                    )

    def test_circuit_has_four_qubits(self):
        """Verify the circuit operates on exactly 4 qubits."""
        qc = four_qubit_grover("0000")
        self.assertEqual(qc.num_qubits, 4)

    def test_invalid_solution_strings_rejected(self):
        """Verify that invalid inputs raise appropriate errors."""
        with self.assertRaises(ValueError):
            four_qubit_grover("010")

        with self.assertRaises(ValueError):
            four_qubit_grover("01010")

        with self.assertRaises(ValueError):
            four_qubit_grover("0120")

        with self.assertRaises(ValueError):
            four_qubit_grover("ABCD")

    def test_iteration_count_is_optimal(self):
        """
        Verify the circuit uses the optimal number of iterations.

        For 4 qubits: floor(pi/4 * sqrt(16)) = 3
        """
        qc = four_qubit_grover("0000")
        total_gates = len(qc.data)

        # Should be between 70 and 90 gates for 3 iterations
        self.assertGreater(total_gates, 70, "Too few gates - maybe only 2 iterations?")
        self.assertLess(total_gates, 90, "Too many gates - maybe 4 iterations?")


if __name__ == "__main__":
    unittest.main(verbosity=2)
