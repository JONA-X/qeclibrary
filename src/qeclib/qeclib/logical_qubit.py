from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

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
    logical_x: List[int]
    logical_z: List[int]

    def _number_of_data_qubits(self) -> int:
        """Returns the number of data qubits involved in this logical qubit.

        Returns:
            int: Number of data qubits
        """
        data_qubit_indices = []
        for stab in self.stabilizers:
            for qb in stab.data_qubits:
                if qb not in data_qubit_indices:
                    data_qubit_indices.append(qb)
        return len(data_qubit_indices)

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

        list_pauli_strs = [stab.get_global_pauli_string(n) for stab in self.stabilizers]

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
