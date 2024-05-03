from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from .logical_qubit import LogicalQubit

CircuitList = List[Tuple[str, List[Union[int, Tuple[int, int]]]]]

op_names: List[str] = [
    "R",
    "X",
    "Y",
    "Z",
    "H",
    "CX",
    "CY",
    "CZ",
]
internal_op_to_stim_map: Dict[str, str] = {
    "R": "R",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "H": "H",
    "CX": "CX",
    "CY": "CY",
    "CZ": "CZ",
}


@dataclass()
class Circuit:
    name: str
    logical_qubits: List[LogicalQubit]
    _circuit: Optional[CircuitList] = Field(default_factory=lambda: [])

    def print_logical_qubits(self):
        for qb in self.logical_qubits:
            print(qb.id)

    def is_log_qb_id_contained(self, id: str):
        for qb in self.logical_qubits:
            if id == qb.id:
                return True
        return False

    def add_logical_qubit(self, logical_qubit: LogicalQubit):
        if self.is_log_qb_id_contained(id):
            raise ValueError("Logical qubit id already exists on the processor")

        self.logical_qubits.append(logical_qubit)

    def _log_qb_valid_check(self, log_qb: LogicalQubit) -> True:
        if log_qb not in self.logical_qubits:
            raise ValueError("Logical qubit does not exist in this circuit.")
        if log_qb.exists == False:
            raise ValueError(
                "Logical qubit does not exist anymore due to a previous merge or split."
            )
        return True

    def x(self, log_qb: LogicalQubit) -> None:
        if self._log_qb_valid_check(log_qb):
            self._circuit += log_qb.x()

    def z(self, log_qb: LogicalQubit) -> None:
        if self._log_qb_valid_check(log_qb):
            self._circuit += log_qb.z()

    def init(self, log_qb: LogicalQubit, state: Union[str, int]) -> None:
        """

        Parameters
        ----------
        state: Union[str, int]
            Must be in [0, 1, '0', '1', '+', '-']

        Returns
        -------
        """
        if self._log_qb_valid_check(log_qb):
            self._circuit += log_qb.init(state)

    def convert_to_stim(self) -> str:
        stim_circ = ""
        # Define coordinates of logical qubits
        for log_qb in self.logical_qubits:
            for id, coords in log_qb.dqb_coords.items():
                stim_circ += f"QUBIT_COORDS({coords[0]}, {coords[1]}) {id}\n"

        # Operations of the circuit
        for op in self._circuit:
            stim_circ += internal_op_to_stim_map[op[0]]
            for qb in op[1]:
                stim_circ += f" {qb}"
            stim_circ += "\n"
        return stim_circ
