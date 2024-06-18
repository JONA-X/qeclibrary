from typing import Literal
from abc import ABC, abstractmethod
from pydantic import Field
from pydantic.dataclasses import dataclass
import copy

from .pauli_op import PauliOp
from .stabilizer import Stabilizer

import numpy as np

CircuitList = list[tuple[str, list[int | tuple[int, int]]]]
Qubit = tuple[int, ...]
import pprint

@dataclass()
class LogicalQubit(ABC):
    """Class representing one logical qubit on a code patch.
    Defining multiple logical qubits inside the same code patch is not
    supported.
    """

    id: str
    stabilizers: list[Stabilizer] | None = None
    log_x: PauliOp | None = None
    log_z: PauliOp | None = None
    exists: bool = (
        True  # Will be set to False once the qubit is merged with another qubit or split into qubits
    )
    log_x_corrections: list[tuple[str, int]] = Field(
        default_factory=lambda: [],
        init=False,
    )
    log_z_corrections: list[tuple[str, int]] = Field(
        default_factory=lambda: [],
        init=False,
    )
    logical_readouts: dict[str, tuple[str, list[tuple[str, int]]]] = Field(
        default_factory=lambda: {},
        init=False,
    )
    circ: object = None

    def __post_init__(self) -> None:
        self._check_correctness()

    def __setattr__(self, prop, val):
        if prop == "id":
            raise AttributeError("You cannot change the id of a logical qubit.")
        super().__setattr__(prop, val)

    def _get_data_qubits(self) -> list[int | tuple[int, int]]:
        data_qubit_indices = []
        for stab in self.stabilizers:
            for qb in stab.pauli_op.data_qubits:
                if qb not in data_qubit_indices:
                    data_qubit_indices.append(qb)
        return data_qubit_indices

    def _get_ancilla_qubits(self) -> list[int | tuple[int, int]]:
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
            print("- The number of stabilizers is not correct :(")
            print(
                f"  There should be {n-1} stabilizers while there are {'only ' if len(self.stabilizers) < n - 1 else ''}{len(self.stabilizers)}"
            )

        list_paulis = [stab.pauli_op for stab in self.stabilizers]

        # Check that stabilizers commute
        list_paulis_to_check = list_paulis
        stabs_commute = True
        while len(list_paulis_to_check) > 0:
            next_pauli = list_paulis_to_check.pop()
            for pauli in list_paulis_to_check:
                if not next_pauli.commutes_with(pauli):
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
            for pauli in list_paulis:
                if not log_op.commutes_with(pauli):
                    logical_operators_commute = False
                    break
        if logical_operators_commute:
            print("+ The logical X and Z operators commute with all stabilizers :)")
        else:
            print(
                "- The logical X and Z operators do not commute with all stabilizers :("
            )

        # Check that logical operators anticommute
        if not self.log_x.commutes_with(self.log_z):
            print("+ The logical X and Z operators anticommute :)")
        else:
            print("- The logical X and Z operators do not anticommute :(")

    def get_neighbour_dqbs(self, qb: int) -> list[int]:
        dqbs = self._get_data_qubits()
        neighbors = [qb for qb in self.circ.get_neighbour_dqbs(qb) if qb in dqbs]
        return neighbors

    def get_dqb_coords(self) -> dict[int, tuple[float, float]]:
        return {index: self.circ.qb_coords[index] for index in self._get_data_qubits()}

    def get_aqb_coords(self) -> dict[int, tuple[float, float]]:
        return {index: self.circ.qb_coords[index] for index in self._get_ancilla_qubits()}

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

    def init(self, state: str | int = 0) -> CircuitList:
        if state not in [0, 1, "0", "1", "+", "-"]:
            raise ValueError(
                "Invalid initial state. Must be in [0, 1, '0', '1', '+', '-']"
            )

        # TODO: Generalize for other cases
        if state in [0, "0"]:
            circ = [("R", self._get_data_qubits())]
            circ += [("Barrier", [])]
            return circ
        elif state in [1, "1"]:
            circ = [("R", self._get_data_qubits())]
            circ += self.x()
            circ += [("Barrier", [])]
            return circ
        elif state in ["+"]:
            circ = [("R", self._get_data_qubits())]
            circ += self.h_trans_raw()
            circ += [("Barrier", [])]
            return circ
        elif state in ["-"]:
            circ = [("R", self._get_data_qubits())]
            circ += self.x()
            circ += self.h_trans_raw()
            circ += [("Barrier", [])]
            return circ
        else:
            raise ValueError("Currently it is only supported to start in |0> or |1>.")

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
        circ_list.append(("H", stab.anc_qubits[0]))
        for i, pauli in enumerate(stab.pauli_op.pauli_string):
            circ_list.append(
                (
                    f"C{pauli}",
                    [
                        stab.anc_qubits[0],
                        stab.pauli_op.data_qubits[i],
                    ],
                )
            )

        circ_list.append(("H", stab.anc_qubits[0]))

        if stab.reset == "reset":
            circ_list.append(("MR", stab.anc_qubits[0]))
        elif stab.reset == "none":
            circ_list.append(("M", stab.anc_qubits[0]))

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
        circuit_list += (
            ("H", [id for id in self._get_ancilla_qubits()]),
            ("Barrier", []),
        )

        for step in range(max_stab_length):
            for stab in self.stabilizers:
                if step < stab.pauli_op.length():
                    circuit_list += (
                        (
                            f"C{stab.pauli_op.pauli_string[step]}",
                            [
                                stab.anc_qubits[0],
                                stab.pauli_op.data_qubits[step],
                            ],
                        ),
                    )
            circuit_list += (("Barrier", []),)

        circuit_list += (
            ("H", [id for id in self._get_ancilla_qubits()]),
            ("Barrier", []),
        )
        stabs_anc_w_reset = []
        stabs_anc_wo_reset = []

        for stab in self.stabilizers:
            if stab.reset == "reset":
                stabs_anc_w_reset.append(stab.anc_qubits[0])
            elif stab.reset == "none":
                stabs_anc_w_reset.append(stab.anc_qubits[0])

        if len(stabs_anc_w_reset) > 0:
            circuit_list += (
                ("MR", [id for id in self._get_ancilla_qubits()]),
                ("Barrier", []),
            )
        if len(stabs_anc_wo_reset) > 0:
            circuit_list += (
                ("M", [id for id in self._get_ancilla_qubits()]),
                ("Barrier", []),
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

    def get_pauli_charges(self):
        pauli_charge_numbers = {}
        for stab in self.stabilizers:
            for i, qb in enumerate(stab.pauli_op.data_qubits):
                if qb in pauli_charge_numbers:
                    pauli_charge_numbers[qb][stab.pauli_op.pauli_string[i]] += 1
                else:
                    pauli_charge_numbers[qb] = {"X": 0, "Y": 0, "Z": 0}
                    pauli_charge_numbers[qb][stab.pauli_op.pauli_string[i]] = 1

        pauli_charges = {}
        for qb, charge in pauli_charge_numbers.items():
            charge_list = (charge["X"] % 2, charge["Y"] % 2, charge["Z"] % 2)
            if charge_list in [(0, 0, 0), (1, 1, 1)]:
                pauli_charges[qb] = "I"
            elif charge_list in [(1, 0, 0), (0, 1, 1)]:
                pauli_charges[qb] = "X"
            elif charge_list in [(0, 1, 0), (1, 0, 1)]:
                pauli_charges[qb] = "Y"
            else:
                pauli_charges[qb] = "Z"
        return pauli_charges

    @abstractmethod
    def split(
        self, split_qbs: list[tuple[int, ...]], new_ids: tuple[str, str]
    ) -> tuple[CircuitList, "LogicalQubit", "LogicalQubit", str, list, list]:
        pass


@dataclass()
class RotSurfCode(LogicalQubit):
    """Class representing one logical qubit on a code patch.
    Defining multiple logical qubit inside the same code patch is not
    supported.
    """

    d: int | None = (
        None  # Code distance along both axes.
        # If d is provided, dx and dz cannot be provided separately
    )
    dx: int | None = None  # Number of columns, minimum length of the X operator
    dz: int | None = None  # Number of rows, minimum length of the Z operator
    # So X operator goes from left to right
    # Z operator goes trom top to bottom

    def __deepcopy__(self, memo):
        new_qb = RotSurfCode(
            id=self.id,
            exists=self.exists,
            circ=None,
            d=self.d,
            dx=self.dx,
            dz=self.dz,
        )
        setattr(new_qb, "stabilizers", copy.deepcopy(self.stabilizers))
        setattr(new_qb, "log_x", copy.deepcopy(self.log_x))
        setattr(new_qb, "log_z", copy.deepcopy(self.log_z))
        setattr(new_qb, "log_x_corrections", copy.deepcopy(self.log_x_corrections))
        setattr(new_qb, "log_z_corrections", copy.deepcopy(self.log_z_corrections))
        setattr(new_qb, "logical_readouts", copy.deepcopy(self.logical_readouts))
        return new_qb

    def __post_init__(self) -> None:
        if self.d is not None:
            if self.dx is not None or self.dz is not None:
                raise ValueError(
                    "If d is provided, dx and dz cannot be provided separately."
                )
            self.dx = self.d
            self.dz = self.d
        else:
            if self.dx is None or self.dz is None:
                raise ValueError("Please specify either both dx and dz, or only d.")

        if self.stabilizers is None:
            stabs = []

            ## XXXX stabilizers
            for row in range(self.dz - 1):
                for col in range(self.dx - 1):
                    if (row + col) % 2 == 1:
                        pauli_str = "XXXX"
                        stabs.append(
                            Stabilizer(
                                pauli_op=PauliOp(
                                    pauli_string=pauli_str,
                                    data_qubits=[
                                        (col, row, 0),
                                        (col + 1, row, 0),
                                        (col, row + 1, 0),
                                        (col + 1, row + 1, 0),
                                    ],
                                ),
                                anc_qubits=[(col+1, row+1, 1)],
                            )
                        )

            ## XX stabilizers
            for row in range(self.dz - 1):
                if row % 2 == 0:
                    stabs.append(
                        Stabilizer(
                            pauli_op=PauliOp(
                                pauli_string="XX",
                                data_qubits=[
                                    (0, row, 0),
                                    (0, row + 1, 0),
                                ],
                            ),
                            anc_qubits=[(0, row+1, 1)],
                        )
                    )
                else:
                    stabs.append(
                        Stabilizer(
                            pauli_op=PauliOp(
                                pauli_string="XX",
                                data_qubits=[
                                    (self.dx - 1, row, 0),
                                    (self.dx - 1, row + 1, 0),
                                ],
                            ),
                            anc_qubits=[(self.dx, row+1, 1)],
                        )
                    )

            ## ZZZZ stabilizers
            for row in range(self.dz - 1):
                for col in range(self.dx - 1):
                    if (row + col) % 2 == 0:
                        pauli_str = "ZZZZ"
                        stabs.append(
                            Stabilizer(
                                pauli_op=PauliOp(
                                    pauli_string=pauli_str,
                                    data_qubits=[
                                        (col, row, 0),
                                        (col + 1, row, 0),
                                        (col, row + 1, 0),
                                        (col + 1, row + 1, 0),
                                    ],
                                ),
                                anc_qubits=[(col+1, row+1, 1)],
                            )
                        )

            ## ZZ stabilizers
            for col in range(self.dx - 1):
                if col % 2 == 1:
                    stabs.append(
                        Stabilizer(
                            pauli_op=PauliOp(
                                pauli_string="ZZ",
                                data_qubits=[
                                    (col, 0, 0),
                                    (col + 1, 0, 0),
                                ],
                            ),
                            anc_qubits=[(col+1, 0, 1)],
                        )
                    )
                else:
                    stabs.append(
                        Stabilizer(
                            pauli_op=PauliOp(
                                pauli_string="ZZ",
                                data_qubits=[
                                    (col, self.dz - 1, 0),
                                    (col + 1, self.dz - 1, 0),
                                ],
                            ),
                            anc_qubits=[(col+1, self.dz, 1)],
                        )
                    )

            self.stabilizers = stabs
            self.log_x = PauliOp(
                pauli_string="X" * self.dx,
                data_qubits=[(i, 0, 0) for i in range(self.dx)]
            )
            self.log_z = PauliOp(pauli_string="Z" * self.dz,
                data_qubits=[(0, i, 0) for i in range(self.dz)])

    def get_def_log_op(self, basis: str) -> PauliOp:
        # Idea: Start at one logical corner (i.e. with Pauli charge Y) and move along
        # an X/Z boundary (specified by basis argument) until reaching another logical corner
        pauli_charges = self.get_pauli_charges()
        for qb, charge in pauli_charges.items():
            if charge == "Y":
                start_qb = qb
                break
        current_qb = start_qb
        log_op_dqbs = [start_qb]
        already_visited = [start_qb]
        completed = False
        while not completed:
            for next_qb in self.get_neighbour_dqbs(current_qb):
                print(next_qb)
                if next_qb in already_visited:
                    continue

                if pauli_charges[next_qb] == "Y":
                    log_op_dqbs.append(next_qb)
                    completed = True
                    break
                elif pauli_charges[next_qb] == basis:
                    log_op_dqbs.append(next_qb)
                    already_visited.append(next_qb)
                    current_qb = next_qb
                    break

        return PauliOp(pauli_string=basis * len(log_op_dqbs), data_qubits=log_op_dqbs)

    def get_def_log_x(self) -> PauliOp:
        return self.get_def_log_op("X")

    def get_def_log_z(self) -> PauliOp:
        return self.get_def_log_op("Z")

    def split(
        self, split_qbs: list[Qubit], new_ids: tuple[str, str]
    ) -> tuple[CircuitList, LogicalQubit, LogicalQubit, str]:
        threshold = 1e-3

        def get_splitted_op(split_qbs: list[Qubit]) -> str:
            """Returns `X` if the logical X operator is split, `Z` if the logical Z
            operator is split."""
            if len(set(split_qbs).intersection(set(self.log_x.data_qubits))) == 0:
                x_op_split = False  # The logical x operator is not split
            else:
                x_op_split = True  # The logical x operator is split

            if len(set(split_qbs).intersection(set(self.log_z.data_qubits))) == 0:
                z_op_split = False  # The logical z operator is not split
            else:
                z_op_split = True  # The logical z operator is split

            if x_op_split and z_op_split:
                raise RuntimeError(
                    "Both the logical x and logical z operator cross the split region. "
                    "This should not happen."
                )
            elif not x_op_split and not z_op_split:
                raise RuntimeError(
                    "Neither the logical x nor the logical z operator cross the split "
                    "region. This should not happen."
                )

            if x_op_split:
                return "X"
            else:
                return "Z"

        def find_stabs_from_dqbs(dqbs: set[Qubit], splitted_op: str) -> list[Stabilizer]:
            stabs = []
            for stab in self.stabilizers:
                intersection_qbs = set(stab.pauli_op.data_qubits) & dqbs
                if set(stab.pauli_op.data_qubits) <= dqbs:
                    stabs.append(stab)
                elif len(intersection_qbs) == 2:
                    # Now consider stabilizers which were weight-4 stabilizers before the split
                    # but now there are only 2 qubits remaining in the new set.
                    # Note that the case with only 1 qubit left are stabilizers which were
                    # previously weight-2 stabilizers at the edge and one of the data qubits
                    # was measured during the split. These stabilizers are discarded during
                    # the split.
                    if stab.pauli_op.pauli_string[0] != splitted_op:
                        continue
                    # This stabilizer has to be modified to exclude the measured split qubits
                    new_pauli_str = ""
                    new_data_qbs = []
                    for i, dqb in enumerate(stab.pauli_op.data_qubits):
                        if dqb in intersection_qbs:
                            new_pauli_str += stab.pauli_op.pauli_string[i]
                            new_data_qbs.append(dqb)
                    new_stab = Stabilizer(
                        pauli_op=PauliOp(
                            pauli_string=new_pauli_str, data_qubits=new_data_qbs
                        ),
                        anc_qubits=stab.anc_qubits,
                        reset=stab.reset,
                    )
                    stabs.append(new_stab)
            return stabs

        def construct_new_log_qb(new_stabs, new_id, splitted_op: str):
            dqbs_new = {
                dqb for stab in new_stabs for dqb in stab.pauli_op.data_qubits
            }
            dqb_coords_new = {qb: self.circ.qb_coords[qb] for qb in dqbs_new}
            new_dx = (
                1
                + max([dqb_coords_new[qb][0] for qb in dqbs_new])
                - min([dqb_coords_new[qb][0] for qb in dqbs_new])
            )
            new_dz = (
                1
                + max([dqb_coords_new[qb][1] for qb in dqbs_new])
                - min([dqb_coords_new[qb][1] for qb in dqbs_new])
            )

            new_log_qb = RotSurfCode(
                new_id,
                stabilizers=new_stabs,
                dx=new_dx,
                dz=new_dz,
                circ=self.circ,  # Associate them with the same circuit
            )

            def get_log_op_intersection(op: PauliOp, dqb_set: list[Qubit]) -> PauliOp:
                new_op_qbs = [
                    qb
                    for qb in op.data_qubits
                    if qb in dqb_set
                ]
                new_op_paulis = [
                    op.pauli_string[i]
                    for i, qb in enumerate(op.data_qubits)
                    if qb in dqb_set
                ]
                new_op_pauli_str = "".join(new_op_paulis)
                return PauliOp(
                    pauli_string=new_op_pauli_str, data_qubits=new_op_qbs
                )

            if splitted_op == "Z":
                new_log_qb.log_z = get_log_op_intersection(self.log_z, dqbs_new)
                new_log_qb.log_x = new_log_qb.get_def_log_x()
            elif splitted_op == "X":
                new_log_qb.log_x = get_log_op_intersection(self.log_x, dqbs_new)
                new_log_qb.log_z = new_log_qb.get_def_log_z()

            return new_log_qb

        if not set(split_qbs) <= set(self._get_data_qubits()):
            raise ValueError(f"The split qubits are not contained in the set of data qubits of this logical qubit. Split qubits: {split_qbs}")

        dqbs_wo_split_qbs = set(self._get_data_qubits()) - set(split_qbs)
        dqbs_log_qb1 = set(self.circ.get_connected_dqbs_in_set(list(dqbs_wo_split_qbs)[0], dqbs_wo_split_qbs))
        # Check whether the two sets contain the same qubits (must be ==, cannot be is)
        if dqbs_log_qb1 == dqbs_wo_split_qbs:
            raise ValueError(f"The split qubits do not separate the logical qubit patch into two patches. Check again the list of split qubits and consider adding additional split qubits. Split qubits: {split_qbs}")

        dqbs_log_qb2 = dqbs_wo_split_qbs - dqbs_log_qb1

        splitted_op = get_splitted_op(split_qbs)
        stabs_log_qb1 = find_stabs_from_dqbs(dqbs_log_qb1, splitted_op)
        stabs_log_qb2 = find_stabs_from_dqbs(dqbs_log_qb2, splitted_op)
        new_qb1 = construct_new_log_qb(stabs_log_qb1, new_ids[0], splitted_op)
        new_qb2 = construct_new_log_qb(stabs_log_qb2, new_ids[1], splitted_op)
        pprint.pp(new_qb1)
        pprint.pp(new_qb2)

        split_qbs_mmt_circ = []
        if splitted_op == "X":
            split_qbs_mmt_circ += [
                ["H", split_qbs],
            ]
        split_qbs_mmt_circ += [
            ["M", split_qbs],
        ]
        log_op_update_stabs1 = []
        log_op_update_stabs2 = []

        return (
            split_qbs_mmt_circ,
            new_qb1,
            new_qb2,
            splitted_op,
            log_op_update_stabs1,
            log_op_update_stabs2,
        )


        def construct_new_log_qb(new_stabs, new_id):
                dqbs_new = {
                    dqb for stab in new_stabs for dqb in stab.pauli_op.data_qubits
                }
                dqb_coords_new
                new_dx = (
                    1
                    + max([dqb_coords_new[qb][0] for qb in dqbs_new])
                    - min([dqb_coords_new[qb][0] for qb in dqbs_new])
                )
                new_dz = (
                    1
                    + max([dqb_coords_new[qb][1] for qb in dqbs_new])
                    - min([dqb_coords_new[qb][1] for qb in dqbs_new])
                )

                new_log_qb = RotSurfCode(
                    new_id,
                    stabilizers=new_stabs,
                    dx=new_dx,
                    dz=new_dz,
                    circ=self.circ,  # Associate them with the same circuit
                )

                if (
                    split_operator == "Z"
                ):  # Z operator is split and X operator is not split
                    if len(
                        set(dqb_coords_new).intersection(set(self.log_x.data_qubits))
                    ) == len(self.log_x.data_qubits):
                        # Logical X is contained on this patch. So just take the old
                        # logical X operator.
                        new_log_qb.log_x = self.log_x
                    elif (
                        len(
                            set(dqb_coords_new).intersection(
                                set(self.log_x.data_qubits)
                            )
                        )
                        == 0
                    ):
                        # Logical X is contained on the other patch. So find a new valid
                        # logical X operator.
                        new_log_qb.log_x = new_log_qb.get_def_log_x()
                    else:
                        raise RuntimeError(
                            f"The logical X operator is not split but it has length {len(self.log_x.data_qubits)} while {len(set(dqb_coords_new).intersection(set(self.log_x.data_qubits)))} of its qubits are contained on the new patch."
                        )

                    new_log_z_pauli_str = ""
                    new_log_z_qbs = []
                    for i, qb in enumerate(self.log_z.data_qubits):
                        if qb in dqbs_new:
                            new_log_z_pauli_str += self.log_z.pauli_string[i]
                            new_log_z_qbs.append(qb)
                    new_log_qb.log_z = PauliOp(
                        pauli_string=new_log_z_pauli_str, data_qubits=new_log_z_qbs
                    )

                elif (
                    split_operator == "X"
                ):  # X operator is split and Z operator is not split
                    if len(
                        set(dqb_coords_new).intersection(set(self.log_z.data_qubits))
                    ) == len(self.log_z.data_qubits):
                        # Logical Z is contained on this patch. So just take the old
                        # logical Z operator.
                        new_log_qb.log_z = self.log_z
                    elif (
                        len(
                            set(dqb_coords_new).intersection(
                                set(self.log_z.data_qubits)
                            )
                        )
                        == 0
                    ):
                        # Logical Z is contained on the other patch. So find a new valid
                        # logical Z operator.
                        new_log_qb.log_z = new_log_qb.get_def_log_z()
                    else:
                        raise RuntimeError(
                            f"The logical Z operator is not split but it has length {len(self.log_z.data_qubits)} while {len(set(dqb_coords_new).intersection(set(self.log_z.data_qubits)))} of its qubits are contained on the new patch."
                        )

                    if len(
                        set(dqb_coords_new).intersection(set(self.log_z.data_qubits))
                    ) == len(self.log_z.data_qubits):
                        new_log_qb.log_z = self.log_z
                    elif (
                        len(
                            set(dqb_coords_new).intersection(
                                set(self.log_z.data_qubits)
                            )
                        )
                        == 0
                    ):
                        new_log_qb.log_z = new_log_qb.get_def_log_z()
                    else:
                        raise RuntimeError(
                            f"The logical Z operator is not split but it has length {len(self.log_z.data_qubits)} while {len(set(dqb_coords_new).intersection(set(self.log_z.data_qubits)))} qubits are contained on the new patch."
                        )

                    new_log_x_pauli_str = ""
                    new_log_x_qbs = []
                    for i, qb in enumerate(self.log_x.data_qubits):
                        if qb in dqbs_new:
                            new_log_x_pauli_str += self.log_x.pauli_string[i]
                            new_log_x_qbs.append(qb)
                    new_log_qb.log_x = PauliOp(
                        pauli_string=new_log_x_pauli_str, data_qubits=new_log_x_qbs
                    )

                return new_log_qb

        new_qb1 = construct_new_log_qb(new_stabs_left, new_ids[0])
        new_qb2 = construct_new_log_qb(new_stabs_right, new_ids[1])
        log_op_update_stabs1 = []
        for stab in boundary_stabs_left:
            if list(set(stab.pauli_op.pauli_string))[0] != split_operator:
                log_op_update_stabs1.append(self.stabilizers.index(stab))
        log_op_update_stabs2 = []
        for stab in boundary_stabs_right:
            if list(set(stab.pauli_op.pauli_string))[0] != split_operator:
                log_op_update_stabs2.append(self.stabilizers.index(stab))


    def shrink(
        self,
        num_rows: int,
        direction: Literal["t", "b", "l", "r"],
    ) -> CircuitList:
        """_summary_

        Parameters
        ----------
        num_rows : int
            _description_
        direction : Literal["t", "b", "l", "r"]
            Direction from where rows/columns should be deleted.

        Returns
        -------
        CircuitList
            _description_
        """
        circuit_list = []
        if direction == "r":
            if num_rows > self.dz - 1:
                raise ValueError(
                    f"Cannot remove {num_rows} columns since there are only {self.dz} "
                    "columns. We need at least one column remaining after shrinking."
                )

            # qubits_to_measure = []

        elif direction in ["t", "b", "l"]:
            raise NotImplementedError(
                "Shrinking in this direction is not yet supported."
            )
        else:
            raise ValueError("Invalid direction. Must be in ['t', 'b', 'l', 'r']")

        return circuit_list
