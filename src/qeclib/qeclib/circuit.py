from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass
import uuid
import itertools
import copy

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
    "M",
    "MR",
    "Barrier",
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
    "M": "M",
    "MR": "MR",
    "Barrier": "TICK",
}


@dataclass()
class Circuit:
    name: str
    logical_qubits: List[LogicalQubit] = Field(init=False, default_factory=lambda: [])
    _circuit: Optional[CircuitList] = Field(default_factory=lambda: [])
    _m_list: List[Tuple[int, int, str, str, Union[None, str]]] = Field(
        default_factory=lambda: []
    )
    _num_measurements: int = 0

    def print_logical_qubits(self):
        for qb in self.logical_qubits:
            print(qb.id)

    def is_log_qb_id_contained(self, id: str):
        for qb in self.logical_qubits:
            if id == qb.id:
                return True
        return False

    def add_logical_qubit(self, logical_qubit: LogicalQubit):
        if self.is_log_qb_id_contained(logical_qubit.id):
            raise ValueError("Logical qubit ID already exists on the processor")

        self.logical_qubits.append(logical_qubit)

    def _log_qb_valid_check(self, log_qb: LogicalQubit) -> True:
        if log_qb not in self.logical_qubits:
            raise ValueError("Logical qubit does not exist in this circuit.")
        if log_qb.exists == False:
            raise ValueError(
                "Logical qubit does not exist anymore due to a previous merge or split."
            )
        return True

    def add_mmt(self, length: int, label: str = None, log_qb_id: str = None) -> str:
        m_id = str(uuid.uuid4())
        if label in [None, ""]:
            label = m_id

        for mmt in self._m_list:
            if mmt[2] == label:
                raise ValueError("Measurement label already exists in the circuit.")

        self._m_list += [(self._num_measurements, length, label, m_id, log_qb_id)]
        self._num_measurements += length
        return m_id

    def x(self, log_qb: LogicalQubit) -> None:
        if self._log_qb_valid_check(log_qb):
            self._circuit += log_qb.x()

    def y(self, log_qb: LogicalQubit) -> None:
        if self._log_qb_valid_check(log_qb):
            self._circuit += log_qb.y()

    def z(self, log_qb: LogicalQubit) -> None:
        if self._log_qb_valid_check(log_qb):
            self._circuit += log_qb.z()

    def h_trans_raw(self, log_qb: LogicalQubit) -> None:
        if self._log_qb_valid_check(log_qb):
            self._circuit += log_qb.h_trans_raw()

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

            for id, coords in log_qb.aqb_coords.items():
                stim_circ += f"QUBIT_COORDS({coords[0]}, {coords[1]}) {log_qb.get_anc_absolute_idx(id)}\n"

        # Operations of the circuit
        for op in self._circuit:
            stim_circ += internal_op_to_stim_map[op[0]]
            if type(op[1]) == int:
                stim_circ += f" {op[1]}"
            else:
                for qb in op[1]:
                    stim_circ += f" {qb}"
            stim_circ += "\n"
        return stim_circ

    def add_def_syndrome_extraction_circuit(
        self, log_qbs: List[LogicalQubit] = []
    ) -> None:
        if len(log_qbs) == 0:
            log_qbs = self.logical_qubits

        uuids = []
        for log_qb in log_qbs:
            self._circuit += log_qb.get_def_syndrome_extraction_circuit()
            m_id = self.add_mmt(len(log_qb.stabilizers), "", log_qb.id)
            uuids.append(m_id)

        return uuids

    def add_par_def_syndrome_extraction_circuit(
        self, log_qbs: List[LogicalQubit] = []
    ) -> None:
        if len(log_qbs) == 0:
            log_qbs = self.logical_qubits

        uuids = []
        for log_qb in log_qbs:
            self._circuit += log_qb.get_par_def_syndrome_extraction_circuit()
            m_id = self.add_mmt(len(log_qb.stabilizers), "", log_qb.id)
            uuids.append(m_id)

        return uuids

    def m_log(self, log_qb: LogicalQubit, basis: str, label: str = "") -> str:
        if basis == "X":
            m_circ = log_qb.m_log_x()
            n = log_qb.log_x.length()
        elif basis == "Y":
            m_circ = log_qb.m_log_y()
            n = log_qb.log_y.length()
        elif basis == "Z":
            m_circ = log_qb.m_log_z()
            n = log_qb.log_z.length()
        else:
            raise ValueError("Invalid basis. Must be in ['X', 'Y', 'Z']")

        self._circuit += m_circ
        m_id = self.add_mmt(n, label, log_qb.id)
        return m_id

    def m_log_x(self, log_qb: LogicalQubit, label: str = "") -> str:
        return self.m_log(log_qb, "X", label)

    def m_log_y(self, log_qb: LogicalQubit, label: str = "") -> str:
        return self.m_log(log_qb, "Y", label)

    def m_log_z(self, log_qb: LogicalQubit, label: str = "") -> str:
        return self.m_log(log_qb, "Z", label)

    def log_QST(self, log_qbs: List[LogicalQubit]) -> List[Tuple[str, Circuit]]:
        if type(log_qbs) != list:
            log_qbs = [log_qbs]

        # bases = list(itertools.product(['X', 'Y', 'Z'], repeat=len(log_qbs)))
        bases = list(itertools.product(["X", "Z"], repeat=len(log_qbs)))
        list_circuits = []
        for basis in bases:
            new_circ = copy.deepcopy(self)
            for i, log_qb in enumerate(log_qbs):
                new_circ.m_log(log_qb, basis[i], f"QST_{log_qb.id}")
            list_circuits.append(("".join(basis), new_circ))
        return list_circuits

    def dict_m_labels_to_res(self, measurements):
        res = {}
        for mmt in self._m_list:
            label = mmt[2]
            res[label] = measurements[mmt[0] : mmt[0] + mmt[1]]
        return res

    def dict_m_uuids_to_res(self, measurements):
        res = {}
        for mmt in self._m_list:
            uuid = mmt[3]
            res[uuid] = measurements[mmt[0] : mmt[0] + mmt[1]]
        return res

    def split(
        self, log_qb: LogicalQubit, split_qbs: List[int], new_ids: Tuple[str, str]
    ) -> List[LogicalQubit]:
        self._log_qb_valid_check(log_qb)

        new_log_qbs = log_qb.split(split_qbs, new_ids)
        return new_log_qbs
