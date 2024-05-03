from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from .pauli_op import *
from .stabilizer import *
from .utilities import *


@dataclass()
class LogicalQubit:
    """Class representing one logical qubit on a code patch.
    Defining multiple logical qubit inside the same code patch is not
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
    )

    def __post_init__(self) -> None:
        self._check_correctness()
        if len(self.dqb_coords) == 0:
            self.create_default_coords()

    def create_default_coords(self):
        for qb in self._get_data_qubits():
            self.dqb_coords[qb] = (qb, 0)

    def _get_data_qubits(self) -> List[Union[int, Tuple[int, int]]]:
        data_qubit_indices = []
        for stab in self.stabilizers:
            for qb in stab.pauli_op.data_qubits:
                if qb not in data_qubit_indices:
                    data_qubit_indices.append(qb)
        return data_qubit_indices

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

    def x(self):
        circuit_list = []
        qubits_dict = self.log_x.get_qubit_groups_for_XYZ()
        for pauli, qbs in qubits_dict.items():
            if len(qbs) > 0:
                circuit_list.append([pauli, qbs])
        return circuit_list

    def z(self):
        circuit_list = []
        qubits_dict = self.log_z.get_qubit_groups_for_XYZ()
        for pauli, qbs in qubits_dict.items():
            if len(qbs) > 0:
                circuit_list.append([pauli, qbs])
        return circuit_list

    def init(self, state: Union[str, int] = 0):
        if state not in [0, 1, "0", "1", "+", "-"]:
            raise ValueError("Invalid state. Must be in [0, 1, '0', '1', '+', '-']")

        # TODO: Generalize for other cases
        if state not in [0, "0"]:
            raise ValueError("Currently it is only supported to start in |0>.")

        return [["R", self._get_data_qubits()]]
