from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass


@dataclass()
class PauliOp:
    pauli_string: str  # String consisting of 'I', 'X', 'Y', 'Z' characters
    data_qubits: List[int]  # Involved data qubit indices. List must have the
    # same length as pauli_string

    def __post_init__(self) -> None:
        if len(self.pauli_string) != len(self.data_qubits):
            raise ValueError(
                "Length of pauli_string must be equal to length of data_qubits"
            )

    def __str__(self) -> str:
        str_repr = ""
        for i, qb in enumerate(self.data_qubits):
            str_repr += f"{self.pauli_string[i]}_{qb}"
        return str_repr

    def get_latex_repr(self) -> str:
        str_repr = ""
        for i, qb in enumerate(self.data_qubits):
            str_repr += f"{self.pauli_string[i]}_{{{qb}}}"
        return str_repr

    def get_global_pauli_string(self, max_id: int) -> str:
        pauli_str = ""
        for i in range(max_id):
            if i in self.data_qubits:
                pauli_str += self.pauli_string[self.data_qubits.index(i)]
            else:
                pauli_str += "I"

        return pauli_str

    def get_qubit_groups_for_XYZ(self):
        x_qubits = []
        y_qubits = []
        z_qubits = []
        for i, pauli in enumerate(self.pauli_string):
            if pauli == "X":
                x_qubits.append(self.data_qubits[i])
            elif pauli == "Y":
                y_qubits.append(self.data_qubits[i])
            elif pauli == "Z":
                z_qubits.append(self.data_qubits[i])

        return {
            "X": x_qubits,
            "Y": y_qubits,
            "Z": z_qubits,
        }

    def length(self):
        length = 0
        for pauli in self.pauli_string:
            if pauli != "I":
                length += 1
        return length

    def commutes_with(self, other_pauli) -> bool:
        """Return True if the provided Pauli operator commutes with this Pauli operator
        and False otherwise."""
        do_commute = True
        for i in range(max(max(self.data_qubits), max(other_pauli.data_qubits)) + 1):
            if i in self.data_qubits and i in other_pauli.data_qubits:
                if self.pauli_string[self.data_qubits.index(i)] != "I" and other_pauli.pauli_string[
                    other_pauli.data_qubits.index(i)] != "I":
                    if self.pauli_string[self.data_qubits.index(i)] != other_pauli.pauli_string[
                        other_pauli.data_qubits.index(i)]:
                        do_commute = not do_commute
        return do_commute
