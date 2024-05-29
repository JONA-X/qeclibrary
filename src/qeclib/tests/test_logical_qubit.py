import unittest

from qeclib import RotSurfCode


class TestRotSurfCode(unittest.TestCase):
    def setUp(self):
        self.Q1 = RotSurfCode(
            id="Q1",
            d=3,
        )

    def test_rot_surf_code(self):
        self.assertEqual(self.Q1.id, "Q1")
        self.assertEqual(len(self.Q1.stabilizers), 8)

    def test_get_pauli_charges(self):
        expected_pauli_charges = {
            1: "Z",
            2: "Y",
            4: "I",
            5: "X",
            3: "X",
            6: "Y",
            7: "Z",
            0: "Y",
            8: "Y",
        }
        self.assertEqual(self.Q1.get_pauli_charges(), expected_pauli_charges)

    def test_h_trans_raw(self):
        circ_list = self.Q1.h_trans_raw()
        circ_list_expected = [["H", [0, 1, 2, 3, 4, 5, 6, 7, 8]], ["Barrier", []]]
        # Check that H is the same. Check the sets of qubits since their order might be different
        self.assertEqual(circ_list[0][0], circ_list_expected[0][0])
        self.assertEqual(set(circ_list[0][1]), set(circ_list_expected[0][1]))
        # Check that barrier exists
        self.assertEqual(circ_list[1], circ_list_expected[1])


if __name__ == "__main__":
    unittest.main()
