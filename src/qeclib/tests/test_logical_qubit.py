import unittest

from qeclib import RotSurfCode


class TestRotSurfCode(unittest.TestCase):
    def test_rot_surf_code(self):
        Q1 = RotSurfCode(
            id="Q1",
            d=3,
        )

        self.assertEqual(Q1.id, "Q1")
        self.assertEqual(len(Q1.stabilizers), 8)


if __name__ == "__main__":
    unittest.main()
