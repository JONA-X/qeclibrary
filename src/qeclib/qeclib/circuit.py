from typing import Literal
from abc import ABC, abstractmethod
from pydantic import Field
from pydantic.dataclasses import dataclass
import uuid
import itertools
import copy
import numpy as np

from .logical_qubit import LogicalQubit, RotSurfCode
from .noise_models import NoiseModel
from .measurement import Measurement

CircuitList = list[tuple[str, list[int | tuple[int, int]]]]
Qubit = tuple[int, ...]

op_names: list[str] = [
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
internal_op_to_stim_map: dict[str, str] = {
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

internal_op_to_qasm_str_map: dict[str, str] = {
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
class Circuit(ABC):
    name: str
    log_qbs: dict[str, LogicalQubit] = Field(init=False, default_factory=lambda: {})
    _circuit: CircuitList | None = Field(default_factory=lambda: [])
    _m_list: list[Measurement] = Field(
        default_factory=lambda: []
    )  # Format of tuples: (Start index in measurement list, length (= number of measured qubits), label, measurement id, id of logical qubit that was measured)
    _num_measurements: int = 0
    qb_coords: dict[Qubit, tuple[float, float]] | None = Field(
        default_factory=lambda: {}
    )  # Map from data qubit indices to coordinatess

    @abstractmethod
    def __deepcopy__(self, memo):
        pass

    def print_logical_qubits(self, only_active: bool | None = True):
        for qb_id, qb in self.log_qbs.items():
            if not only_active or qb.exists:
                print(qb.id)

    def exists_log_qb(self, id: str) -> int:
        # Check if activate qubit with the same id exists
        for qb_id, qb in self.log_qbs.items():
            if id == qb_id and qb.exists is True:
                return 1
        # Otherwise check if there is an inactive qubit with the same id
        for qb_id, qb in self.log_qbs.items():
            if qb.exists is False and id == qb.id:
                return 2
        return 0

    @abstractmethod
    def add_logical_qubit(
        self, logical_qubit: LogicalQubit, start_pos: tuple[int, int] = (0, 0)
    ):
        pass

    def remove_logical_qubit(self, logical_qubit_id: str) -> bool:
        if self.exists_log_qb(logical_qubit_id) == 0:
            raise ValueError(
                f"Logical qubit with id '{logical_qubit_id}' does not exist on the processor and cannot be deleted."
            )
        elif self.exists_log_qb(logical_qubit_id) == 2:
            raise ValueError(
                f"There has been a logical qubit with name '{logical_qubit_id}' once but it has already been deleted. Cannot delete it again."
            )

        for qb_id, qb in self.log_qbs.items():
            if qb_id == logical_qubit_id:
                qb.exists = False  # Mark the original logical qubit as non-existent
                return True

        return False

    def log_qb_id_valid_check(self, log_qb_id: str) -> bool:
        if not isinstance(log_qb_id, str):
            raise ValueError("Logical qubit id must be a string.")

        log_qb = self.exists_log_qb(log_qb_id)
        if log_qb == 0:
            raise ValueError(
                f"Logical qubit id '{log_qb_id}' does not exist in this circuit."
            )
        if log_qb == 2:
            raise ValueError(
                f"Logical qubit with id '{log_qb_id}' does not exist anymore due to a previous merge or split."
            )
        return True

    def add_mmt(
        self,
        number_of_mmts: int,
        label: str = None,
        log_qb_id: str = None,
        type: str = None,
        related_obj: str = None,
    ) -> str:
        m_id = str(uuid.uuid4())
        if label in [None, ""]:
            label = m_id

        for mmt in self._m_list:
            if mmt.label == label:
                raise ValueError(
                    f"Measurement label '{label}' already exists. Labels must be unique."
                )

        self._m_list.append(
            Measurement(
                self._num_measurements,
                number_of_mmts,
                label,
                log_qb_id,
                m_id,
                type,
                related_obj,
            )
        )
        self._num_measurements += number_of_mmts
        return m_id

    def x(self, log_qb_id: str) -> None:
        if self.log_qb_id_valid_check(log_qb_id):
            self._circuit += self.log_qbs[log_qb_id].x()

    def y(self, log_qb_id: str) -> None:
        if self.log_qb_id_valid_check(log_qb_id):
            self._circuit += self.log_qbs[log_qb_id].y()

    def z(self, log_qb_id: str) -> None:
        if self.log_qb_id_valid_check(log_qb_id):
            self._circuit += self.log_qbs[log_qb_id].z()

    def h_trans_raw(self, log_qb_id: str) -> None:
        if self.log_qb_id_valid_check(log_qb_id):
            self._circuit += self.log_qbs[log_qb_id].h_trans_raw()

    def init(self, log_qb_id: str, state: str | int) -> None:
        """

        Parameters
        ----------
        state: Union[str, int]
            Must be in [0, 1, '0', '1', '+', '-']

        Returns
        -------
        """
        if self.log_qb_id_valid_check(log_qb_id):
            self._circuit += self.log_qbs[log_qb_id].init(state)

    def convert_to_stim(self, noise_model: NoiseModel = None) -> str:
        dqb_keys = list(self.qb_coords.keys())
        stim_circ = ""
        # Define coordinates of logical qubits
        for id, coords in self.qb_coords.items():
            stim_circ += f"QUBIT_COORDS({coords[0]}, {coords[1]}) {dqb_keys.index(id)}\n"

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
                    stim_circ += f" {dqb_keys.index(qb)}"
            stim_circ += "\n"
        return stim_circ

    def convert_to_qasm(self) -> str:
        qasm_str = "OPENQASM 3;"
        qasm_str += 'include "stdgates.inc";'

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
        self, log_qbs: list[LogicalQubit] = []
    ) -> None:
        if len(log_qbs) == 0:
            log_qbs = self.log_qbs

        uuids = []
        for log_qb in log_qbs:
            self._circuit += log_qb.get_def_syndrome_extraction_circuit()
            m_ids = [
                self.add_mmt(1, "", log_qb.id, "stabilizer", str(stab.pauli_op))
                for stab in log_qb.stabilizers
            ]
            uuids += m_ids

        return uuids

    def add_par_def_syndrome_extraction_circuit(
        self,
        log_qb_id: str,
        label: str | None = None,
    ) -> list[str]:
        self.log_qb_id_valid_check(
            log_qb_id
        )  # Raise exception if the provided logical qubit id is not valid

        uuids = []
        self._circuit += self.log_qbs[
            log_qb_id
        ].get_par_def_syndrome_extraction_circuit()
        for i, stab in enumerate(self.log_qbs[log_qb_id].stabilizers):
            if label is not None:
                m_label = label + str(stab.pauli_op)
            else:
                m_label = None  # Pass None to the function, so that it will use the uuid as a label
            m_id = self.add_mmt(1, m_label, log_qb_id, "stabilizer", str(stab.pauli_op))
            uuids.append(m_id)

        return uuids

    def add_par_def_syndrome_extraction_circuit_all_log_qbs(
        self, round: int = None
    ) -> list[str]:
        all_uuids = []
        for log_qb_id, log_qb in self.log_qbs.items():
            if round is not None:
                label = f"QEC_r{round}_" + log_qb_id
            else:
                label = None
            if log_qb.exists:
                uuids = self.add_par_def_syndrome_extraction_circuit(log_qb_id, label)
                all_uuids += uuids
        return all_uuids

    def m_log(self, log_qb_id: str, basis: str, label: str = "") -> str:
        if not isinstance(log_qb_id, str):
            raise ValueError("Logical qubit id must be a string")
        self.log_qb_id_valid_check(
            log_qb_id
        )  # Raise exception if the provided logical qubit id is not valid

        if basis == "X":
            m_circ = self.log_qbs[log_qb_id].m_log_x()
            n = self.log_qbs[log_qb_id].log_x.length()
            corrections_list = self.log_qbs[log_qb_id].log_x_corrections
            self.log_qbs[log_qb_id].log_x_corrections = []  # Clear the corrections list
        elif basis == "Y":
            m_circ = self.log_qbs[log_qb_id].m_log_y()
            n = self.log_qbs[log_qb_id].log_y.length()
            corrections_list = (
                self.log_qbs[log_qb_id].log_x_corrections
                + self.log_qbs[log_qb_id].log_z_corrections
            )  # Do both corrections
            self.log_qbs[log_qb_id].log_x_corrections = []  # Clear the corrections list
            self.log_qbs[log_qb_id].log_z_corrections = []  # Clear the corrections list
        elif basis == "Z":
            m_circ = self.log_qbs[log_qb_id].m_log_z()
            n = self.log_qbs[log_qb_id].log_z.length()
            corrections_list = self.log_qbs[log_qb_id].log_z_corrections
            self.log_qbs[log_qb_id].log_z_corrections = []  # Clear the corrections list
        else:
            raise ValueError("Invalid basis. Must be in ['X', 'Y', 'Z']")

        self._circuit += m_circ

        m_id = self.add_mmt(n, label, log_qb_id, "log_op", log_qb_id)
        self.log_qbs[log_qb_id].logical_readouts[m_id] = (
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

    def log_QST(
        self, log_qbs: list[str] = None, bases: list[str] = ["X", "Y", "Z"]
    ) -> list[tuple[str, "Circuit"]]:
        if log_qbs is None:
            log_qbs = [qb_id for qb_id, qb in self.log_qbs.items() if qb.exists is True]

        if not isinstance(log_qbs, list):
            log_qbs = [log_qbs]

        bases = list(itertools.product(bases, repeat=len(log_qbs)))
        list_circuits = []
        for basis in bases:
            basis_str = "".join(basis)
            new_circ = copy.deepcopy(self)
            for i, log_qb in enumerate(log_qbs):
                if basis[i] != "I":
                    new_circ.m_log(log_qb, basis[i], f"QST_{log_qb}_{basis[i]}")
            list_circuits.append((basis_str, new_circ))
        return list_circuits

    def dict_m_labels_to_res(self, measurements):
        res = {}
        for mmt in self._m_list:
            res[mmt.label] = measurements[mmt.index : mmt.index + mmt.number_of_mmts]
        return res

    def dict_m_labels_to_uuids(self):
        res = {}
        for mmt in self._m_list:
            res[mmt.label] = mmt.uuid
        return res

    def dict_m_uuids_to_labels(self):
        res = {}
        for mmt in self._m_list:
            res[mmt.uuid] = mmt.label
        return res

    def dict_m_uuids_to_res(self, measurements):
        res = {}
        for mmt in self._m_list:
            res[mmt.uuid] = measurements[mmt.index : mmt.index + mmt.number_of_mmts]
        return res

    def get_log_dqb_readout(self, measurements, m_id, log_qb_id: str) -> int:
        val = np.sum(self.dict_m_uuids_to_res(measurements)[m_id]) % 2
        mmt_tuple = self.log_qbs[log_qb_id].logical_readouts[m_id]
        for corr in mmt_tuple[1]:
            val += self.dict_m_uuids_to_res(measurements)[corr[0]][corr[1]]
        return val % 2

    def split(
        self,
        log_qb_id: str,
        split_qbs: list[int],
        new_ids: tuple[str, str],
    ):
        self.log_qb_id_valid_check(log_qb_id)

        qec_uuids = self.add_par_def_syndrome_extraction_circuit(log_qb_id)

        (
            split_circ,
            new_log_qb1,
            new_log_qb2,
            split_operator,
            log_op_update_stabs1,
            log_op_update_stabs2,
        ) = self.log_qbs[log_qb_id].split(split_qbs, new_ids)

        self._circuit += split_circ
        m_id = self.add_mmt(len(split_qbs), "", log_qb_id, "split", log_qb_id)

        if split_operator == "X":
            measured_split_qb = list(
                set(split_qbs).intersection(
                    set(self.log_qbs[log_qb_id].log_x.data_qubits)
                )
            )[0]
            new_log_qb1.log_x_corrections.append(
                (m_id, split_qbs.index(measured_split_qb))
            )  # Arbitrary choice of correcting the first qubit. We could have done the
            # same to qubit 2 instead.

            # Correct both Z_L operators
            for id in log_op_update_stabs1:
                new_log_qb1.log_z_corrections.append((qec_uuids[id], 0))
            for id in log_op_update_stabs2:
                new_log_qb2.log_z_corrections.append((qec_uuids[id], 0))

        elif split_operator == "Z":
            measured_split_qb = list(
                set(split_qbs).intersection(
                    set(self.log_qbs[log_qb_id].log_z.data_qubits)
                )
            )[0]
            new_log_qb1.log_z_corrections.append(
                (m_id, split_qbs.index(measured_split_qb))
            )  # Arbitrary choice of correcting the first qubit. We could have done the
            # same to qubit 2 instead.

            # TODO: Add the corrections for X_L
            raise NotImplementedError(
                "Splitting the Z operator is not yet implemented."
            )

        self.remove_logical_qubit(log_qb_id)
        self.log_qbs[new_log_qb1.id] = new_log_qb1
        self.log_qbs[new_log_qb2.id] = new_log_qb2

    def shrink(
        self,
        log_qb_id: str,
        num_rows: int,
        direction: Literal["t", "b", "l", "r"],
    ):
        raise NotImplementedError("Shrink is not yet supported")
        self.log_qb_id_valid_check(log_qb_id)

        shrink_circ = self.log_qbs[log_qb_id].shrink(num_rows, direction)
        self._circuit += shrink_circ

    def grow(
        self,
        log_qb_id: str,
        num_rows: int,
        direction: Literal["t", "b", "l", "r"],
    ):
        self.log_qb_id_valid_check(log_qb_id)

        grow_circ = self.log_qbs[log_qb_id].grow(num_rows, direction)
        self._circuit += grow_circ

    def get_connected_dqbs_in_set(self, starting_qubit: Qubit, qubit_set: set[Qubit]) -> set[Qubit]:
        queue = self.get_neighbour_dqbs(starting_qubit)
        neighbors = []
        while len(queue) > 0:
            current_qb = queue.pop(0)
            for next_qb in self.get_neighbour_dqbs(current_qb):
                if (
                    (next_qb not in neighbors)
                    and (next_qb in qubit_set)
                    and next_qb is not starting_qubit
                    ):
                    neighbors.append(next_qb)
                    queue.append(next_qb)
        return neighbors


@dataclass
class SquareLattice(Circuit):
    rows: int = None
    cols: int = None

    def __init__(self, rows: int = None, cols: int = None):
        if rows is None:
            raise ValueError("Number of rows must be provided.")
        else:
            self.rows = rows

        if cols is None:
            raise ValueError("Number of columns must be provided.")
        else:
            self.cols = cols

    def __post_init__(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.qb_coords[(c, r, 0)] = (c, r)

        # Ancilla qubits
        for r in range(self.rows+1):
            for c in range(self.cols+1):
                self.qb_coords[(c, r, 1)] = (c - 0.5, r - 0.5) # Ancilla qubit

    def __deepcopy__(self, memo):
        new_circ = SquareLattice(name=self.name, rows=self.rows, cols=self.cols)
        log_qbs = copy.deepcopy(self.log_qbs)
        for log_qb_id, log_qb in log_qbs.items():
            log_qb.circ = new_circ
        new_circ.log_qbs = log_qbs
        new_circ._circuit = copy.deepcopy(self._circuit)
        new_circ._m_list = copy.deepcopy(self._m_list)
        new_circ._num_measurements = self._num_measurements
        new_circ.qb_coords = copy.deepcopy(self.qb_coords)
        return new_circ

    @property
    def dqb_coords(self):
        return {index: coords for index, coords in self.qb_coords.items() if index[2] == 0}

    @property
    def aqb_coords(self):
        return {index: coords for index, coords in self.qb_coords.items() if index[2] == 1}

    def get_qb_coords(self, qb: Qubit) -> tuple[float, float]:
        if qb not in self.qb_coords:
            raise ValueError(f"Qubit {qb} does not exist on the processor.")
        return self.qb_coords[qb]

    def get_neighbour_dqbs(self, qb: Qubit) -> list[Qubit]:
        neighbours = []
        for c, r in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (
                qb[0] + c >= 0
                and qb[0] + c < self.cols
                and qb[1] + r >= 0
                and qb[1] + r < self.rows
            ):
                neighbours.append((qb[0] + c, qb[1] + r, 0))

        return neighbours

    def add_logical_qubit(
        self, logical_qubit: LogicalQubit, start_pos: tuple[int, int] = (0, 0)
    ):
        if isinstance(logical_qubit, RotSurfCode):
            if any([not isinstance(start_pos[i], int) for i in range(2)]):
                raise ValueError("Start position must be a tuple of integers.")
            if any([start_pos[i] < 0 for i in range(2)]):
                raise ValueError(
                    "Start position coordinates must be larger or equal to 1."
                )
            if start_pos[0] + logical_qubit.dx > self.cols:
                raise ValueError(
                    "Start position column is too large."
                )
            if start_pos[1] + logical_qubit.dz > self.rows:
                raise ValueError(
                    "Start position row is too large."
                )

            if (
                logical_qubit.id not in self.log_qbs
                or self.log_qbs[logical_qubit.id].exists is False
            ):
                self.log_qbs[logical_qubit.id] = logical_qubit
                logical_qubit.circ = self
                for stab in logical_qubit.stabilizers:
                    for i, qb in enumerate(stab.pauli_op.data_qubits):
                        stab.pauli_op.data_qubits[i] = (
                            stab.pauli_op.data_qubits[i][0] + start_pos[0],
                            stab.pauli_op.data_qubits[i][1] + start_pos[1],
                            0
                        )
                    for i, qb in enumerate(stab.anc_qubits):
                        stab.anc_qubits[i] = (
                            stab.anc_qubits[i][0] + start_pos[0],
                            stab.anc_qubits[i][1] + start_pos[1],
                            1
                        )

                for i, qb in enumerate(logical_qubit.log_x.data_qubits):
                    logical_qubit.log_x.data_qubits[i] = (
                        logical_qubit.log_x.data_qubits[i][0] + start_pos[0],
                        logical_qubit.log_x.data_qubits[i][1] + start_pos[1],
                        0
                    )
                for i, qb in enumerate(logical_qubit.log_z.data_qubits):
                    logical_qubit.log_z.data_qubits[i] = (
                        logical_qubit.log_z.data_qubits[i][0] + start_pos[0],
                        logical_qubit.log_z.data_qubits[i][1] + start_pos[1],
                        0
                    )
            else:
                raise ValueError("Logical qubit already exists.")
        else:
            raise NotImplementedError(
                "Only RotSurfCode logical qubits are supported on the square lattice for now."
            )
