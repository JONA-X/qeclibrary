import unittest

from qeclib.math import (
    stab_to_F2,
)


class TestMath(unittest.TestCase):
    def test_stab_to_F2(self):
        stab_matrix = stab_to_F2(
            [
                ["XXI", [0, 1, None], 42],
                ["IXX", [None, 1, 2], 42],
                ["ZZZ", [0, 1, 2], 42],
            ]
        )
        expected_stab_matrix = [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ]
        for i in range(len(stab_matrix)):
            self.assertEqual(tuple(stab_matrix[i]), tuple(expected_stab_matrix[i]))


if __name__ == "__main__":
    unittest.main()
