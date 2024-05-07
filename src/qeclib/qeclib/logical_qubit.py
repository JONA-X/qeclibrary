from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from .pauli_op import *
from .stabilizer import *
from .utilities import *

import pprint

CircuitList = List[Tuple[str, List[Union[int, Tuple[int, int]]]]]


@dataclass()
class LogicalQubit:
    """Class representing one logical qubit on a code patch.
    Defining multiple logical qubits inside the same code patch is not
    supported.
    """

    id: str
    stabilizers: List[Stabilizer]
    log_x: PauliOp
    log_z: PauliOp
    exists: bool = (
        True  # Will be set to False once the qubit is merged with another qubit or split into qubits
    )
    dqb_coords: Optional[Dict[int, Tuple[float, float]]] = Field(
        default_factory=lambda: {}
    )  # Coordinates of the data qubits
    aqb_coords: Optional[Dict[int, Tuple[float, float]]] = Field(
        default_factory=lambda: {}
    )  # Coordinates of the ancilla qubits

    def __post_init__(self) -> None:
        self._check_correctness()
        if len(self.dqb_coords) == 0:
            self.create_default_dqb_coords()
        if len(self.aqb_coords) == 0:
            self.create_default_aqb_coords()

    def __setattr__(self, prop, val):
        if prop == "id":
            raise AttributeError("You cannot change the ID of a logical qubit.")
        super().__setattr__(prop, val)

    def create_default_dqb_coords(self):
        for qb in self._get_data_qubits():
            self.dqb_coords[qb] = (qb, 0)

    def create_default_aqb_coords(self):
        for qb in self._get_ancilla_qubits():
            self.dqb_coords[qb] = (qb, 0)

    def _get_data_qubits(self) -> List[Union[int, Tuple[int, int]]]:
        data_qubit_indices = []
        for stab in self.stabilizers:
            for qb in stab.pauli_op.data_qubits:
                if qb not in data_qubit_indices:
                    data_qubit_indices.append(qb)
        return data_qubit_indices

    def _get_ancilla_qubits(self) -> List[Union[int, Tuple[int, int]]]:
        ancilla_qubit_indices = []
        for stab in self.stabilizers:
            for qb in stab.anc_qubits:
                if qb not in ancilla_qubit_indices:
                    ancilla_qubit_indices.append(qb)
        return ancilla_qubit_indices

    def _number_of_data_qubits(self) -> int:
        """Returns the number of data qubits involved in this logical qubit.

        Returns:
            int: Number of data qubits
        """
        return len(self._get_data_qubits())

    def _check_correctness(self) -> bool:
        """Checks whether this class instance define a valid single logical
        qubit. Checks performed:
        - number of stabilizers == number of data qubits - 1
        - all stabilizers commute
        - X_L and Z_L both commute with all stabilizers
        - X_L and Z_L anticommute

        Returns:
            bool: True if this object specifies one valid logical qubit
        """
        n = self._number_of_data_qubits()

        # Check number of stabilizers
        if n - 1 == len(self.stabilizers):
            print("+ The number of stabilizers is correct :)")
        else:
            print(f"- The number of stabilizers is not correct :(")
            print(
                f"  There should be {n-1} stabilizers while there are {'only ' if len(self.stabilizers) < n - 1 else ''}{len(self.stabilizers)}"
            )

        list_pauli_strs = [
            stab.pauli_op.get_global_pauli_string(n) for stab in self.stabilizers
        ]

        # Check that stabilizers commute
        list_pauli_strs_to_check = list_pauli_strs
        stabs_commute = True
        while len(list_pauli_strs_to_check) > 0:
            next_pauli = list_pauli_strs_to_check.pop()
            for pauli in list_pauli_strs_to_check:
                if not check_commutation_of_pauli_string(next_pauli, pauli):
                    stabs_commute = False
                    break
        if stabs_commute:
            print("+ The stabilizers all commute :)")
        else:
            print("- The stabilizers don't commute :(")

        # Check that logical operators commute with stabilizers
        logical_operators_commute = True
        log_ops = [self.log_x, self.log_z]
        for log_op in log_ops:
            for pauli in list_pauli_strs:
                if not check_commutation_of_pauli_string(log_op, pauli):
                    logical_operators_commute = False
                    break
        if logical_operators_commute:
            print("+ The logical X and Z operators commute with all stabilizers :)")
        else:
            print(
                "- The logical X and Z operators do not commute with all stabilizers :("
            )

        # Check that logical operators anticommute
        if not check_commutation_of_pauli_string(
            self.log_x.get_global_pauli_string(n), self.log_z.get_global_pauli_string(n)
        ):
            print("+ The logical X and Z operators anticommute :)")
        else:
            print("- The logical X and Z operators do not anticommute :(")

    def x(self) -> CircuitList:
        circuit_list = []
        qubits_dict = self.log_x.get_qubit_groups_for_XYZ()
        for pauli, qbs in qubits_dict.items():
            if len(qbs) > 0:
                circuit_list.append([pauli, qbs])
        circuit_list += [["Barrier", []]]
        return circuit_list

    def y(self) -> CircuitList:
        raise NotImplementedError("Measurement of Y operator is not yet supported.")

    def z(self) -> CircuitList:
        circuit_list = []
        qubits_dict = self.log_z.get_qubit_groups_for_XYZ()
        for pauli, qbs in qubits_dict.items():
            if len(qbs) > 0:
                circuit_list.append([pauli, qbs])
        circuit_list += [["Barrier", []]]
        return circuit_list

    def h_trans_raw(self) -> CircuitList:
        circuit_list = [["H", self._get_data_qubits()]]
        circuit_list += [["Barrier", []]]
        return circuit_list

    def init(self, state: Union[str, int] = 0) -> CircuitList:
        if state not in [0, 1, "0", "1", "+", "-"]:
            raise ValueError("Invalid state. Must be in [0, 1, '0', '1', '+', '-']")

        # TODO: Generalize for other cases
        if state in [0, "0"]:
            circ = [["R", self._get_data_qubits()]]
            circ += [["Barrier", []]]
            return circ
        elif state in [1, "1"]:
            circ = [["R", self._get_data_qubits()]]
            circ += self.x()
            circ += [["Barrier", []]]
            return circ
        else:
            raise ValueError("Currently it is only supported to start in |0> or |1>.")

    def get_anc_absolute_idx(self, anc_idx):
        return anc_idx + max(self._get_data_qubits()) + 1

    def get_stab_def_circuit(self, stab: Stabilizer) -> CircuitList:
        if len(stab.anc_qubits) > 1:
            raise ValueError(
                "More than one ancilla qubit per stabilizer is currently not supported."
            )
        if stab.reset == "conditional":
            raise ValueError("Conditional reset is not yet supported")
        if stab.reset not in ["reset", "none"]:
            raise ValueError("Invalid argument for `reset`")

        circ_list: CircuitList = []
        circ_list.append(("H", self.get_anc_absolute_idx(stab.anc_qubits[0])))
        for i, pauli in enumerate(stab.pauli_op.pauli_string):
            circ_list.append(
                (
                    f"C{pauli}",
                    [
                        self.get_anc_absolute_idx(stab.anc_qubits[0]),
                        stab.pauli_op.data_qubits[i],
                    ],
                )
            )

        circ_list.append(("H", self.get_anc_absolute_idx(stab.anc_qubits[0])))

        if stab.reset == "reset":
            circ_list.append(("MR", self.get_anc_absolute_idx(stab.anc_qubits[0])))
        elif stab.reset == "none":
            circ_list.append(("M", self.get_anc_absolute_idx(stab.anc_qubits[0])))

        return circ_list

    def get_def_syndrome_extraction_circuit(self):
        circuit_list = []
        for stab in self.stabilizers:
            circuit_list += self.get_stab_def_circuit(stab)
        return circuit_list

    def get_par_def_syndrome_extraction_circuit(self):
        max_stab_length = 0
        for stab in self.stabilizers:
            max_stab_length = max(max_stab_length, stab.pauli_op.length())

        circuit_list = []
        circuit_list.append(
            ("H", [self.get_anc_absolute_idx(id) for id in self._get_ancilla_qubits()])
        )

        for step in range(max_stab_length):
            for stab in self.stabilizers:
                if step < stab.pauli_op.length():
                    circuit_list.append(
                        (
                            f"C{stab.pauli_op.pauli_string[step]}",
                            [
                                self.get_anc_absolute_idx(stab.anc_qubits[0]),
                                stab.pauli_op.data_qubits[step],
                            ],
                        )
                    )

        circuit_list.append(
            ("H", [self.get_anc_absolute_idx(id) for id in self._get_ancilla_qubits()])
        )
        circuit_list.append(
            ("M", [self.get_anc_absolute_idx(id) for id in self._get_ancilla_qubits()])
        )
        return circuit_list

    def m_log_x(self) -> CircuitList:
        m_circ = [
            ["H", self.log_x.data_qubits],
            ["M", self.log_x.data_qubits],
        ]
        return m_circ

    def m_log_y(self) -> CircuitList:
        return []
        # raise NotImplementedError("Measurement of Y operator is not yet supported.")

    def m_log_z(self) -> CircuitList:
        m_circ = [
            ["M", self.log_z.data_qubits],
        ]
        return m_circ


@dataclass()
class RotSurfCode(LogicalQubit):
    """Class representing one logical qubit on a code patch.
    Defining multiple logical qubit inside the same code patch is not
    supported.
    """

    def transversal_h(self) -> CircuitList:
        circuit_list = []
        for qbs in self._get_data_qubits():
            circuit_list.append(["H", qbs])
        return circuit_list

    def split(self, split_qbs: List[int], new_ids: Tuple[str, str]):
        split_direction_hor = True
        split_direction_vert = True
        for qb in split_qbs:
            if self.dqb_coords[qb][0] != self.dqb_coords[split_qbs[0]][0]:
                split_direction_hor = False
            if self.dqb_coords[qb][1] != self.dqb_coords[split_qbs[0]][1]:
                split_direction_vert = False

        if split_direction_hor and split_direction_vert:
            raise ValueError(
                "Split qubits must be either in the same row or in the same column."
            )
        elif not split_direction_hor and not split_direction_vert:
            raise ValueError(
                "Split qubits must be either in the same row or in the same column."
            )
        elif split_direction_hor:
            split_direction = "hor"  # Horizontal
            cut_x = self.dqb_coords[split_qbs[0]][0]
        else:
            split_direction = "ver"  # Vertical
            cut_y = self.dqb_coords[split_qbs[0]][1]

        if split_direction == "hor":
            new_stabs_left = []
            for stab in self.stabilizers:
                all_dqbs_left = True
                for dqb in stab.pauli_op.data_qubits:
                    if self.dqb_coords[dqb][0] > cut_x:
                        all_dqbs_left = False
                        break
                if all_dqbs_left:
                    new_stabs_left.append(stab)

            new_stabs_right = []
            for stab in self.stabilizers:
                all_dqbs_right = True
                for dqb in stab.pauli_op.data_qubits:
                    if self.dqb_coords[dqb][0] < cut_x:
                        all_dqbs_right = False
                        break
                if all_dqbs_right:
                    new_stabs_right.append(stab)

            pprint.pp(new_stabs_right)
            # new_qb1 = RotSurfCode(
            #     new_ids[0],
            #     stabilizers=new_stabs_left,
            #     log_x=PauliOp(pauli_string="X" * dx, data_qubits=list(np.arange(dx) * dz)),
            #     log_z=PauliOp(pauli_string="Z" * dz, data_qubits=range(dz)),
            #     dqb_coords=dqb_coords,
            #     aqb_coords=aqb_coords,
            # )

        split_qbs_mmt_circ = [
            ["M", split_qbs],
        ]

        return split_qbs_mmt_circ
