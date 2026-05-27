"""
test_gates.py
=============
Unit tests for individual quantum gate components.

These tests verify that each building block works correctly in isolation.
Testing components separately makes debugging easier.

TESTING APPROACH:
----------------
1. Test each gate function independently
2. Use small circuits where we can calculate expected results by hand
3. Verify both the gate sequence and the mathematical effect
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.gates import (
    apply_hadamard_all,
    apply_pauli_x_all,
    apply_multi_controlled_z,
    apply_oracle_marker,
    apply_diffusion_operator
)
from src.utils import validate_solution_string, calculate_optimal_iterations


class TestUtilityFunctions(unittest.TestCase):
    """Tests for helper functions in utils.py."""

    def test_validate_valid_solutions(self):
        """Valid 4-bit strings should pass."""
        valid = ["0000", "1111", "0010", "1010"]
        for s in valid:
            with self.subTest(solution=s):
                validate_solution_string(s)

    def test_validate_invalid_solutions(self):
        """Invalid strings should raise ValueError."""
        invalid = ["010", "01010", "0120", "ABCD", ""]
        for s in invalid:
            with self.subTest(solution=s):
                with self.assertRaises(ValueError):
                    validate_solution_string(s)

    def test_optimal_iterations(self):
        """Verify iteration calculation for different qubit counts."""
        test_cases = {
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 6,
        }
        for n_qubits, expected in test_cases.items():
            with self.subTest(qubits=n_qubits):
                result = calculate_optimal_iterations(n_qubits)
                self.assertEqual(result, expected)


class TestHadamardGate(unittest.TestCase):
    """Tests for Hadamard gate application."""

    def test_single_qubit_superposition(self):
        """
        H|0> should create equal superposition.

        Mathematical result:
            H|0> = (|0> + |1>) / sqrt(2)
            Probabilities: |0> = 50%, |1> = 50%
        """
        qc = QuantumCircuit(1)
        apply_hadamard_all(qc)

        state = Statevector.from_instruction(qc)
        probs = np.abs(state.data) ** 2

        self.assertAlmostEqual(probs[0], 0.5, places=10)
        self.assertAlmostEqual(probs[1], 0.5, places=10)

    def test_all_qubits_get_hadamard(self):
        """Verify H is applied to all qubits when no list provided."""
        qc = QuantumCircuit(3)
        apply_hadamard_all(qc)

        gate_names = [instr.name for instr in qc.data]
        self.assertEqual(gate_names.count("h"), 3)

    def test_specific_qubits(self):
        """Verify we can target specific qubits."""
        qc = QuantumCircuit(3)
        apply_hadamard_all(qc, qubits=[0, 2])

        gate_names = [instr.name for instr in qc.data]
        self.assertEqual(gate_names.count("h"), 2)


class TestPauliXGate(unittest.TestCase):
    """Tests for Pauli-X (NOT) gate application."""

    def test_flip_single_qubit(self):
        """
        X|0> should flip to |1>.

        Mathematical result:
            X|0> = |1>
            Probability of |1> = 100%
        """
        qc = QuantumCircuit(1)
        apply_pauli_x_all(qc)

        state = Statevector.from_instruction(qc)
        probs = np.abs(state.data) ** 2

        self.assertAlmostEqual(probs[0], 0.0, places=10)
        self.assertAlmostEqual(probs[1], 1.0, places=10)

    def test_all_qubits_get_x(self):
        """Verify X is applied to all qubits when no list provided."""
        qc = QuantumCircuit(4)
        apply_pauli_x_all(qc)

        gate_names = [instr.name for instr in qc.data]
        self.assertEqual(gate_names.count("x"), 4)


class TestMultiControlledZ(unittest.TestCase):
    """Tests for the H-MCX-H multi-controlled-Z construction."""

    def test_mcZ_tags_correct_state(self):
        """
        Verify MCZ applies phase flip to |1111> only.

        Setup: Start in |1111> state
        Apply: MCZ with controls [0,1,2] and target 3
        Result: |1111> should have -1 amplitude (phase flip)
        """
        qc = QuantumCircuit(4)

        # Prepare |1111> state
        for i in range(4):
            qc.x(i)

        # Apply MCZ
        apply_multi_controlled_z(qc, [0, 1, 2], 3)

        state = Statevector.from_instruction(qc)

        # |1111> is index 15
        amplitude = state.data[15]

        # Should have -1 amplitude
        self.assertAlmostEqual(amplitude.real, -1.0, places=10)
        self.assertAlmostEqual(amplitude.imag, 0.0, places=10)

    def test_mcZ_does_not_tag_other_states(self):
        """
        Verify MCZ does NOT affect states where controls aren't all |1>.
        """
        qc = QuantumCircuit(4)

        # Prepare |0111> state (control q0 is |0>)
        for i in range(1, 4):
            qc.x(i)

        # Apply MCZ
        apply_multi_controlled_z(qc, [0, 1, 2], 3)

        state = Statevector.from_instruction(qc)

        # |0111> is index 14
        amplitude = state.data[14]

        # Should still have +1 amplitude
        self.assertAlmostEqual(amplitude.real, 1.0, places=10)

    def test_target_cannot_be_control(self):
        """Verify error when target is in control list."""
        qc = QuantumCircuit(3)
        with self.assertRaises(ValueError):
            apply_multi_controlled_z(qc, [0, 1, 2], 2)


class TestOracleMarker(unittest.TestCase):
    """Tests for the oracle marker function."""

    def test_oracle_tags_solution(self):
        """
        Verify oracle applies -1 phase to the solution state.

        Test with solution "0010":
            q3=0, q2=0, q1=1, q0=0
            Statevector index: reverse("0010") = "0100" = 4
        """
        qc = QuantumCircuit(4)

        # Prepare |0010> state
        qc.x(1)

        # Apply oracle
        apply_oracle_marker(qc, "0010")

        state = Statevector.from_instruction(qc)

        # Check amplitude at index 4
        amplitude = state.data[4]
        self.assertAlmostEqual(amplitude.real, -1.0, places=10)

    def test_oracle_does_not_tag_non_solution(self):
        """Verify non-solution states are unaffected."""
        qc = QuantumCircuit(4)

        # Prepare |0000> state (not the solution)
        # Apply oracle for solution "0010"
        apply_oracle_marker(qc, "0010")

        state = Statevector.from_instruction(qc)

        # |0000> should still have +1 amplitude
        amplitude = state.data[0]
        self.assertAlmostEqual(amplitude.real, 1.0, places=10)


class TestDiffusionOperator(unittest.TestCase):
    """Tests for the diffusion (amplification) operator."""

    def test_diffusion_on_uniform_state(self):
        """
        Test diffusion on a uniform superposition.

        After oracle marks one state, diffusion should:
        - Amplify the marked state
        - Attenuate unmarked states
        """
        qc = QuantumCircuit(2)

        # Create uniform superposition
        apply_hadamard_all(qc)

        # Mark |11> with phase flip (simulate oracle effect)
        qc.cz(0, 1)

        # Apply diffusion
        apply_diffusion_operator(qc)

        state = Statevector.from_instruction(qc)
        probs = np.abs(state.data) ** 2

        # |11> should be amplified
        self.assertGreater(probs[3], 0.9)

        # Other states should be attenuated
        for i in range(3):
            self.assertLess(probs[i], 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
