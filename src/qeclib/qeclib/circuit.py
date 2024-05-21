from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple, Literal
from pydantic import Field
from pydantic.dataclasses import dataclass
import uuid
import itertools
import copy
import numpy as np

from .logical_qubit import LogicalQubit
from .noise_models import NoiseModel

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
    "DEPOLARIZE1": "DEPOLARIZE1",
    "DEPOLARIZE2": "DEPOLARIZE2",
}

internal_op_to_qasm_str_map: Dict[str, str] = {
    "R": "R",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "H": "h",
    "CX": "cnot",
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
    ) # Format of tuples: (Start index in measurement list, length (= number of measured qubits), label, measurement id, id of logical qubit that was measured)
    _m_list: List[Tuple[int, int, str, str, Union[None, str]]] = Field(
        default_factory=lambda: []
    ) # Format of tuples: (Start index in measurement list, length (= number of measured qubits), label, measurement id, id of logical qubit that was measured)
    _num_measurements: int = 0

    def __deepcopy__(self, memo):
        new_circ = Circuit(name=self.name)
        new_circ.logical_qubits = copy.deepcopy(self.logical_qubits)
        new_circ._circuit = copy.deepcopy(self._circuit)
        new_circ._m_list = copy.deepcopy(self._m_list)
        new_circ._num_measurements = copy.deepcopy(self._num_measurements)
        return new_circ

    def print_logical_qubits(self, only_active: Optional[bool] = True):
        for qb in self.logical_qubits:
            if not only_active or qb.exists:
                print(qb.id)

    def exists_log_qb(self, id: str) -> int:
        # Check if activate qubit with the same id exists
        for qb in self.logical_qubits:
            if id == qb.id and qb.exists is True:
                return 1
        # Otherwise check if there is an inactive qubit with the same id
        for qb in self.logical_qubits:
            if qb.exists is False and id == qb.id:
                return 2
        return 0

    def get_log_qb(self, id: str, raise_exceptions: bool = True, raise_warnings: bool = False) -> LogicalQubit:
        if not isinstance(id, str):
            raise ValueError("Logical qubit ID must be a string")

        log_qb = None
        found_qb_but_inactive = False
        for qb in self.logical_qubits:
            if id == qb.id:
                log_qb = qb
                if qb.exists is True:
                    found_qb_but_inactive = False # Need to set back to False in case before we found an inactive qubit with the same id already
                    break
                else:
                    found_qb_but_inactive = True

        if log_qb is None:
            if raise_exceptions:
                raise ValueError("Logical qubit with the given ID does not exist in this circuit.")
            if raise_warnings:
                raise Warning("Logical qubit with the given ID does not exist in this circuit.")
        if found_qb_but_inactive:
            if raise_exceptions or raise_warnings:
                raise Warning("Logical qubit ID does not exist among the active qubits but there has previously been a logical qubit with the same ID.")

        return log_qb

    def add_logical_qubit(self, logical_qubit: LogicalQubit):
        log_qb = self.get_log_qb(logical_qubit.id, raise_exceptions=False, raise_warnings=False)
        if log_qb is None or log_qb.exists is False:
            self.logical_qubits.append(logical_qubit)
        else:
            raise ValueError("Logical qubit already exists.")

    def remove_logical_qubit(self, logical_qubit_id: str) -> bool:
        if self.exists_log_qb(logical_qubit_id) == 0:
            raise ValueError(
                "Logical qubit does not exist on the processor and cannot be deleted."
            )
        elif self.exists_log_qb(logical_qubit_id) == 2:
            raise ValueError(
                f"There has been a logical qubit with name {logical_qubit_id} once but it has already been deleted. Cannot delete it again."
            )

        for i, qb in enumerate(self.logical_qubits):
            if qb.id == logical_qubit_id:
                self.logical_qubits[i].exists = (
                    False  # Mark the original logical qubit as non-existent
                )
                return True

        return False

    def _log_qb_id_valid_check(self, log_qb_id: str) -> bool:
        if not isinstance(log_qb_id, str):
            raise ValueError("Logical qubit ID must be a string")

        log_qb = self.exists_log_qb(log_qb_id)
        if log_qb == 0:
            raise ValueError("Logical qubit does not exist in this circuit.")
        if log_qb == 2:
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

    def x(self, log_qb_id: str) -> None:
        if self._log_qb_id_valid_check(log_qb_id):
            self._circuit += self.get_log_qb(log_qb_id).x()

    def y(self, log_qb_id: str) -> None:
        if self._log_qb_id_valid_check(log_qb_id):
            self._circuit += self.get_log_qb(log_qb_id).y()

    def z(self, log_qb_id: str) -> None:
        if self._log_qb_id_valid_check(log_qb_id):
            self._circuit += self.get_log_qb(log_qb_id).z()

    def h_trans_raw(self, log_qb_id: str) -> None:
        if self._log_qb_id_valid_check(log_qb_id):
            self._circuit += self.get_log_qb(log_qb_id).h_trans_raw()

    def init(self, log_qb: LogicalQubit, state: Union[str, int]) -> None:
        """

        Parameters
        ----------
        state: Union[str, int]
            Must be in [0, 1, '0', '1', '+', '-']

        Returns
        -------
        """
        if self._log_qb_id_valid_check(log_qb.id):
            self._circuit += log_qb.init(state)

    def convert_to_stim(self, noise_model: NoiseModel = None) -> str:
        stim_circ = ""
        # Define coordinates of logical qubits
        for log_qb in self.logical_qubits:
            for id, coords in log_qb.dqb_coords.items():
                stim_circ += f"QUBIT_COORDS({coords[0]}, {coords[1]}) {id}\n"

            for id, coords in log_qb.aqb_coords.items():
                stim_circ += f"QUBIT_COORDS({coords[0]}, {coords[1]}) {id}\n"

        if noise_model is None:
            operation_list = self._circuit
        else:
            operation_list = noise_model.add_errors_to_circuit(self._circuit)

        # Operations of the circuit
        for op in operation_list:
            stim_circ += internal_op_to_stim_map[op[0]]
            if op[0] in ["DEPOLARIZE1", "DEPOLARIZE2"]:
                stim_circ += f"({op[2]})"
            if isinstance(op[1], int):
                stim_circ += f" {op[1]}"
            else:
                for qb in op[1]:
                    stim_circ += f" {qb}"
            stim_circ += "\n"
        return stim_circ

    def convert_to_qasm(self) -> str:
        qasm_str = "OPENQASM 3;"
        qasm_str += "include \"stdgates.inc\";"

        # Define coordinates of logical qubits
        for log_qb in self.logical_qubits:
            for id, coords in log_qb.dqb_coords.items():
                qasm_str += f"QUBIT_COORDS({coords[0]}, {coords[1]}) {id}\n"

            for id, coords in log_qb.aqb_coords.items():
                qasm_str += f"QUBIT_COORDS({coords[0]}, {coords[1]}) {id}\n"

        # Operations of the circuit
        for op in self._circuit:
            qasm_str += internal_op_to_qasm_str_map[op[0]]
            if isinstance(op[1], int):
                qasm_str += f" {op[1]}"
            else:
                for qb in op[1]:
                    qasm_str += f" {qb}"
            qasm_str += "\n"
        return qasm_str

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
        self,
        log_qb_id: str,
        label: Optional[str] = None,
    ) -> List[str]:
        self._log_qb_id_valid_check(log_qb_id) # Raise exception if the provided logical qubit id is not valid
        log_qb = self.get_log_qb(log_qb_id)

        uuids = []
        self._circuit += log_qb.get_par_def_syndrome_extraction_circuit()
        for i, stab in enumerate(log_qb.stabilizers):
            if label is not None:
                m_label = label + str(stab.pauli_op)
            else:
                m_label = None # Pass None to the function, so that it will use the uuid as a label
            m_id = self.add_mmt(1, m_label, log_qb.id)
            uuids.append(m_id)

        return uuids

    def add_par_def_syndrome_extraction_circuit_all_log_qbs(self, round: int = None) -> List[str]:
        all_uuids = []
        for log_qb in self.logical_qubits:
            if round is not None:
                label = f"QEC_r{round}_" + log_qb.id
            else:
                label = None
            uuids = self.add_par_def_syndrome_extraction_circuit(log_qb.id, label)
            all_uuids += uuids
        return all_uuids

    def m_log(self, log_qb_id: str, basis: str, label: str = "") -> str:
        if not isinstance(log_qb_id, str):
            raise ValueError("Logical qubit ID must be a string")
        self._log_qb_id_valid_check(log_qb_id) # Raise exception if the provided logical qubit id is not valid
        log_qb = self.get_log_qb(log_qb_id)

        if basis == "X":
            m_circ = log_qb.m_log_x()
            n = log_qb.log_x.length()
            corrections_list = log_qb.log_x_corrections
            log_qb.log_x_corrections = []  # Clear the corrections list
        elif basis == "Y":
            m_circ = log_qb.m_log_y()
            n = log_qb.log_y.length()
            corrections_list = (
                log_qb.log_x_corrections + log_qb.log_z_corrections
            )  # Do both corrections
            log_qb.log_x_corrections = []  # Clear the corrections list
            log_qb.log_z_corrections = []  # Clear the corrections list
        elif basis == "Z":
            m_circ = log_qb.m_log_z()
            n = log_qb.log_z.length()
            corrections_list = log_qb.log_z_corrections
            log_qb.log_z_corrections = []  # Clear the corrections list
        else:
            raise ValueError("Invalid basis. Must be in ['X', 'Y', 'Z']")

        self._circuit += m_circ
        m_id = self.add_mmt(n, label, log_qb.id)
        log_qb.logical_readouts[m_id] = (
            basis,
            corrections_list,
        )
        return m_id

    def m_log_x(self, log_qb_id: str, label: str = "") -> str:
        return self.m_log(log_qb_id, "X", label)

    def m_log_y(self, log_qb_id: str, label: str = "") -> str:
        return self.m_log(log_qb_id, "Y", label)

    def m_log_z(self, log_qb_id: str, label: str = "") -> str:
        return self.m_log(log_qb_id, "Z", label)

    def log_QST(self, log_qbs: List[str], bases: Optional[List[str]] = ["X", "Y", "Z"]) -> List[Tuple[str, Circuit]]:
        if log_qbs is None:
            log_qbs = [c.id for c in self.logical_qubits if c.exists is True]

        if not isinstance(log_qbs, list):
            log_qbs = [log_qbs]

        bases = list(itertools.product(bases, repeat=len(log_qbs)))
        list_circuits = []
        for basis in bases:
            new_circ = copy.deepcopy(self)
            for i, log_qb in enumerate(log_qbs):
                new_circ.m_log(log_qb, basis[i], f"QST_{log_qb}")
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

    def get_log_dqb_readout(self, measurements, m_id, qb_id: str) -> int:
        val = np.sum(self.dict_m_uuids_to_res(measurements)[m_id]) % 2
        mmt_tuple = self.get_log_qb(qb_id).logical_readouts[m_id]
        for corr in mmt_tuple[1]:
            val += self.dict_m_uuids_to_res(measurements)[corr[0]][corr[1]]
        return val % 2

    def split(
        self,
        logical_qubit_id: str,
        split_qbs: List[int],
        new_ids: Tuple[str, str],
    ) -> Tuple[str, LogicalQubit, LogicalQubit]:
        log_qb = self.get_log_qb(logical_qubit_id)
        self._log_qb_id_valid_check(logical_qubit_id)

        split_circ, new_log_qb1, new_log_qb2, split_operator = log_qb.split(
            split_qbs, new_ids
        )

        self._circuit += split_circ
        m_id = self.add_mmt(len(split_qbs), log_qb_id=log_qb.id)

        if split_operator == "X":
            measured_split_qb = list(
                set(split_qbs).intersection(set(log_qb.log_x.data_qubits))
            )[0]
            new_log_qb1.log_x_corrections.append(
                (m_id, split_qbs.index(measured_split_qb))
            )
            # TODO: Add the corrections for Z_L
        elif split_operator == "Z":
            measured_split_qb = list(
                set(split_qbs).intersection(set(log_qb.log_z.data_qubits))
            )[0]
            new_log_qb1.log_z_corrections.append(
                (m_id, split_qbs.index(measured_split_qb))
            )
            # TODO: Add the corrections for X_L

        self.remove_logical_qubit(logical_qubit_id)
        self.logical_qubits += [new_log_qb1, new_log_qb2]

        return m_id, new_log_qb1, new_log_qb2

    def shrink(
            self,
            logical_qubit_id: str,
            num_rows: int,
            direction: Literal["t", "b", "l", "r"],
    ):
        self._log_qb_id_valid_check(logical_qubit_id)
        log_qb = self.get_log_qb(logical_qubit_id)

        shrink_circ = log_qb.shrink(num_rows, direction)
        self._circuit += shrink_circ