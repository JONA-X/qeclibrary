import unittest
import numpy as np
import warnings

from qeclib.math import (
    commute,
    generate_group,
    stabilizer_distribution,
    MacWilliams,
    normalizer_distribution,
    find_distance,
    operator_set_commute,
    is_valid_tableau,
)


class TestMath(unittest.TestCase):
    def setUp(self):
        # Stabilizer matrix of the [[6,3,2]] code with stabilizers IIXXXX, YYIIYY, ZZZZII
        self.stab_matrix_632 = np.array(
            [
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            ]
        )

        # Stabilizer matrix of the [[12,2,3]] Twisted Toric Code with 5 XXXX and 5 ZZZZ
        # stabilizers
        S_X1 = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        S_X2 = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        S_X3 = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        S_X4 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        S_X5 = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        S_Z1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
        S_Z2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        S_Z3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        S_Z4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
        S_Z5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]
        self.stab_matrix_22_twisted_toric = np.array(
            [S_X1, S_X2, S_X3, S_X4, S_X5, S_Z1, S_Z2, S_Z3, S_Z4, S_Z5]
        )
        log_X1 = [
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        log_Z1 = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
        ]
        log_X2 = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
        ]
        log_Z2 = [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        self.log_ops_22_twisted_toric = [(log_X1, log_Z1), (log_X2, log_Z2)]

        # Stabilizer matrix of the rotated d=3 surface code
        self.stab_matrix_surf17 = np.array(
            [
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            ]
        )
        self.log_ops_surf17 = [
            (
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            )
        ]

    def test_commutation_function(self):
        op1 = [1, 0, 0, 0]  # XI
        op2 = [0, 0, 1, 0]  # ZI
        op3 = [1, 0, 1, 0]  # YI
        op4 = [1, 1, 0, 0]  # XX
        op5 = [0, 0, 0, 1]  # IZ
        op6 = [0, 0, 1, 1]  # ZZ
        op7 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # ZZZZZ
        op8 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # XXXXX
        op9 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # XXXXI
        self.assertEqual(commute(op1, op2), False)
        self.assertEqual(commute(op3, op1), False)
        self.assertEqual(commute(op3, op2), False)
        self.assertEqual(commute(op1, op4), True)
        self.assertEqual(commute(op4, op5), False)
        self.assertEqual(commute(op4, op6), True)
        self.assertEqual(commute(op2, op6), True)
        self.assertEqual(commute(op7, op8), False)
        self.assertEqual(commute(op7, op9), True)
        self.assertEqual(commute(op8, op9), True)

        with self.assertRaises(ValueError) as context:
            commute(op1, op9)
        self.assertTrue(
            "The two operators must have the same length" in str(context.exception)
        )

    def test_generate_group(self):
        basis_vecs = np.array([[1, 0, 0, 0], [1, 1, 0, 0]])
        expected_group = [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        self.assertEqual(
            set(tuple(g) for g in generate_group(basis_vecs)),
            set(tuple(g) for g in expected_group),
        )

        # Test that the generated group is still the same, even when too many basis vectors are given
        basis_vecs = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]])
        self.assertEqual(
            set(tuple(g) for g in generate_group(basis_vecs)),
            set(tuple(g) for g in expected_group),
        )

        # 5 qubit code
        basis_vecs = np.array(
            [
                [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            ]
        )
        expected_group = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        ]
        for stab in expected_group:
            self.assertIn(
                tuple(stab), set(tuple(g) for g in generate_group(basis_vecs))
            )
        self.assertEqual(len(generate_group(basis_vecs)), 16)

    def test_stabilizer_distribution(self):
        # Check the stabilizer distribution of the [[6,3,2]] code with stabilizers
        # IIXXXX, YYIIYY, ZZZZII
        self.assertEqual(
            tuple(stabilizer_distribution(self.stab_matrix_632)), (1, 0, 0, 0, 3, 0, 4)
        )

        # Check the stabilizer distribution of the [[12,2,3]] Twisted Toric Code with
        # 5 XXXX and 5 ZZZZ stabilizers
        self.assertEqual(
            tuple(stabilizer_distribution(self.stab_matrix_22_twisted_toric)),
            (1, 0, 0, 0, 12, 0, 68, 0, 381, 0, 516, 0, 46),
        )

    def test_MacWilliams(self):
        # Given the first Shor-Laflamme distribution for the [[6,3,2]] code, calculate
        # the second distribution using the MacWilliams transform
        A = [1, 0, 0, 0, 3, 0, 4]
        M = MacWilliams(len(A) - 1)
        B = M @ A * 8
        self.assertEqual(tuple(B), (1, 0, 21, 56, 171, 168, 95))

        # Same for the the Surface-7 code
        A = [1, 0, 6, 0, 27, 0, 68, 0, 135, 0, 150, 0, 125]
        M = MacWilliams(len(A) - 1)
        B = np.array(M @ A * 8, dtype=int)
        self.assertEqual(
            tuple(B), (1, 0, 30, 24, 339, 480, 1972, 3024, 6327, 6752, 7566, 4056, 2197)
        )

    def test_normalizer_distribution(self):
        # Check the normalizer distribution of the [[6,3,2]] code with stabilizers
        # IIXXXX, YYIIYY, ZZZZII
        self.assertEqual(
            tuple(normalizer_distribution(self.stab_matrix_632)),
            (1, 0, 21, 56, 171, 168, 95),
        )

        # Check the normalizer distribution of the [[12,2,3]] Twisted Toric Code with
        # 5 XXXX and 5 ZZZZ stabilizers
        self.assertEqual(
            tuple(normalizer_distribution(self.stab_matrix_22_twisted_toric)),
            (1, 0, 0, 16, 45, 144, 776, 1200, 4107, 3248, 4632, 1536, 679),
        )
        # Check that we get the same result when the stabilizer distribution is provided
        self.assertEqual(
            tuple(
                normalizer_distribution(
                    stab_distribution=(1, 0, 0, 0, 12, 0, 68, 0, 381, 0, 516, 0, 46)
                )
            ),
            (1, 0, 0, 16, 45, 144, 776, 1200, 4107, 3248, 4632, 1536, 679),
        )

    def test_find_distance(self):
        self.assertEqual(find_distance(self.stab_matrix_632), 2)
        self.assertEqual(find_distance(self.stab_matrix_22_twisted_toric), 3)
        self.assertEqual(find_distance(self.stab_matrix_surf17), 3)

    def test_operator_set_commute(self):
        # Check that the example stabilizer matrices commute
        self.assertTrue(operator_set_commute(self.stab_matrix_632))
        self.assertTrue(operator_set_commute(self.stab_matrix_22_twisted_toric))
        self.assertTrue(operator_set_commute(self.stab_matrix_surf17))

        # Check that the set of operators does not commute anymore when adding some
        # random other non-commuting operators
        self.assertFalse(
            operator_set_commute(
                np.vstack(
                    (
                        self.stab_matrix_surf17,
                        np.array(
                            [
                                1,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                            ]
                        ),
                    )
                )
            )
        )
        self.assertFalse(
            operator_set_commute(
                np.vstack(
                    (
                        self.stab_matrix_surf17,
                        np.array(
                            [
                                1,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                1,
                                0,
                                0,
                                0,
                                0,
                                0,
                                1,
                                0,
                                0,
                                0,
                            ]
                        ),
                    )
                )
            )
        )

    def test_is_valid_tableau(self):
        # Check that default definition yields a valid tableau
        self.assertTrue(is_valid_tableau(self.stab_matrix_surf17, self.log_ops_surf17))
        # Relabel X_L and Z_L. Should still be a valid tableau
        self.assertTrue(
            is_valid_tableau(
                self.stab_matrix_surf17,
                [
                    (
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    )
                ],
            )
        )
        # Define X and Z the wrong way around, i.e. swap first column vs first row
        self.assertFalse(
            is_valid_tableau(
                self.stab_matrix_surf17,
                [
                    (
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    )
                ],
            )
        )
        # Randomly modify the operators
        self.assertFalse(
            is_valid_tableau(
                self.stab_matrix_surf17,
                [
                    (
                        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    )
                ],
            )
        )
        # Use two different X operators, i.e. the pair does not commute
        self.assertFalse(
            is_valid_tableau(
                self.stab_matrix_surf17,
                [
                    (
                        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    )
                ],
            )
        )

        # Check twisted toric code
        self.assertTrue(
            is_valid_tableau(
                self.stab_matrix_22_twisted_toric, self.log_ops_22_twisted_toric
            )
        )
        # Mix up the logical operators so that the commutation/anticommutation relations are not satisfied
        self.assertFalse(
            is_valid_tableau(
                self.stab_matrix_22_twisted_toric,
                [
                    (
                        self.log_ops_22_twisted_toric[0][0],
                        self.log_ops_22_twisted_toric[1][1],
                    ),
                    (
                        self.log_ops_22_twisted_toric[1][0],
                        self.log_ops_22_twisted_toric[0][1],
                    ),
                ],
            )
        )
        # Add one of the logical operators to the stabilizer list
        self.assertFalse(
            is_valid_tableau(
                np.vstack(
                    (
                        self.stab_matrix_22_twisted_toric,
                        self.log_ops_22_twisted_toric[0][0],
                    )
                ),
                self.log_ops_22_twisted_toric,
            )
        )
        # Check warning if the system is not fully defined:
        # - remove one stabilizer from the list
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            is_valid_tableau(
                self.stab_matrix_22_twisted_toric[:-2], self.log_ops_22_twisted_toric
            )
            self.assertEqual(len(warning_list), 1)
            self.assertIs(warning_list[0].category, UserWarning)
            self.assertEqual(
                str(warning_list[0].message),
                "The system is not fully defined, i.e. m + k < n",
            )
        # - provide only one logical operator pair and not both
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            is_valid_tableau(
                self.stab_matrix_22_twisted_toric, [self.log_ops_22_twisted_toric[0]]
            )
            self.assertEqual(len(warning_list), 1)
            self.assertIs(warning_list[0].category, UserWarning)
            self.assertEqual(
                str(warning_list[0].message),
                "The system is not fully defined, i.e. m + k < n",
            )
        # Check warning if the system is over-defined:
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            is_valid_tableau(
                np.vstack(
                    (
                        self.stab_matrix_22_twisted_toric,
                        self.stab_matrix_22_twisted_toric,
                    )
                ),
                self.log_ops_22_twisted_toric,
            )
            self.assertEqual(len(warning_list), 1)
            self.assertIs(warning_list[0].category, UserWarning)
            self.assertEqual(
                str(warning_list[0].message),
                "The system is over-defined, i.e. m + k > n. The stabilizers or logical operators are not independent.",
            )


if __name__ == "__main__":
    unittest.main()
