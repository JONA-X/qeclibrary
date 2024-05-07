import unittest

import numpy as np
from qeclib import Stabilizer, LogicalQubit, RotSurfCode, PauliOp, Circuit


class TestRotSurfCode(unittest.TestCase):
    def test_rot_surf_code(self):
        dx = 3  # Number of rows, minimum length of the X operator
        dz = 3  # Number of columns, minimum length of the Z operator
        # So left and right boundary are Z boundaries
        # Top and bottom boundary are X boundaries
        stabs = []
        anc_idx = 0
        aqb_coords = {}

        ## ZZZZ stabilizers
        for row in range(dx - 1):
            for col in range(dz - 1):
                if (row + col) % 2 == 1:
                    pauli_str = "ZZZZ"
                    stabs.append(
                        Stabilizer(
                            pauli_op=PauliOp(
                                pauli_string=pauli_str,
                                data_qubits=[
                                    row * dz + col,
                                    row * dz + col + 1,
                                    (row + 1) * dz + col,
                                    (row + 1) * dz + col + 1,
                                ],
                            ),
                            anc_qubits=[anc_idx],
                        )
                    )
                    aqb_coords[anc_idx] = (row + 1.5, col + 1.5)
                    anc_idx += 1

        ## ZZ stabilizers
        for row in range(dx - 1):
            if row % 2 == 0:
                stabs.append(
                    Stabilizer(
                        pauli_op=PauliOp(
                            pauli_string="ZZ",
                            data_qubits=[
                                row * dz,
                                (row + 1) * dz,
                            ],
                        ),
                        anc_qubits=[anc_idx],
                    )
                )
                aqb_coords[anc_idx] = (row + 1.5, 0.5)
                anc_idx += 1
            else:
                stabs.append(
                    Stabilizer(
                        pauli_op=PauliOp(
                            pauli_string="ZZ",
                            data_qubits=[
                                row * dz + dz - 1,
                                (row + 1) * dz + dz - 1,
                            ],
                        ),
                        anc_qubits=[anc_idx],
                    )
                )
                aqb_coords[anc_idx] = (row + 1.5, dz + 0.5)
                anc_idx += 1

        ## XXXX stabilizers
        for row in range(dx - 1):
            for col in range(dz - 1):
                if (row + col) % 2 == 0:
                    pauli_str = "XXXX"
                    stabs.append(
                        Stabilizer(
                            pauli_op=PauliOp(
                                pauli_string=pauli_str,
                                data_qubits=[
                                    row * dz + col,
                                    row * dz + col + 1,
                                    (row + 1) * dz + col,
                                    (row + 1) * dz + col + 1,
                                ],
                            ),
                            anc_qubits=[anc_idx],
                        )
                    )
                    aqb_coords[anc_idx] = (row + 1.5, col + 1.5)
                    anc_idx += 1

        ## XX stabilizers
        for col in range(dz - 1):
            if col % 2 == 1:
                stabs.append(
                    Stabilizer(
                        pauli_op=PauliOp(
                            pauli_string="XX",
                            data_qubits=[
                                col,
                                col + 1,
                            ],
                        ),
                        anc_qubits=[anc_idx],
                    )
                )
                aqb_coords[anc_idx] = (0.5, col + 1.5)
                anc_idx += 1
            else:
                stabs.append(
                    Stabilizer(
                        pauli_op=PauliOp(
                            pauli_string="XX",
                            data_qubits=[
                                (dx - 1) * dz + col,
                                (dx - 1) * dz + col + 1,
                            ],
                        ),
                        anc_qubits=[anc_idx],
                    )
                )
                aqb_coords[anc_idx] = (dx + 0.5, col + 1.5)
                anc_idx += 1

        dqb_coords = {}
        for i in range(dx * dz):
            dqb_coords[i] = (1 + i // dz, 1 + i % dz)

        Q1 = RotSurfCode(
            id="Q1",
            stabilizers=stabs,
            log_x=PauliOp(pauli_string="X" * dx, data_qubits=list(np.arange(dx) * dz)),
            log_z=PauliOp(pauli_string="Z" * dz, data_qubits=range(dz)),
            dqb_coords=dqb_coords,
            aqb_coords=aqb_coords,
        )

        self.assertEqual(Q1.id, "Q1")
        self.assertEqual(len(Q1.stabilizers), 8)


if __name__ == "__main__":
    unittest.main()
