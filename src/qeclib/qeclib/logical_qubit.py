from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple, Literal
from abc import ABC, abstractmethod
from pydantic import Field
from pydantic.dataclasses import dataclass

from .pauli_op import PauliOp
from .stabilizer import Stabilizer

import numpy as np

CircuitList = List[Tuple[str, List[Union[int, Tuple[int, int]]]]]


@dataclass()
class LogicalQubit(ABC):
    """Class representing one logical qubit on a code patch.
    Defining multiple logical qubits inside the same code patch is not
    supported.
    """

    id: str
    stabilizers: Optional[List[Stabilizer]] = None
    log_x: Optional[PauliOp] = None
    log_z: Optional[PauliOp] = None
    exists: bool = (
        True  # Will be set to False once the qubit is merged with another qubit or split into qubits
    )
    dqb_coords: Optional[Dict[int, Tuple[float, float]]] = Field(
        default_factory=lambda: {}
    )  # Coordinates of the data qubits
    aqb_coords: Optional[Dict[int, Tuple[float, float]]] = Field(
        default_factory=lambda: {}
    )  # Coordinates of the ancilla qubits
    log_x_corrections: List[Tuple[str, int]] = Field(
        default_factory=lambda: [],
        init=False,
    )
    log_z_corrections: List[Tuple[str, int]] = Field(
        default_factory=lambda: [],
        init=False,
    )
    logical_readouts: Dict[str, Tuple[str, List[Tuple[str, int]]]] = Field(
        default_factory=lambda: {},
        init=False,
    )
    circ: object = None

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

    def get_neighbour_qbs(self, qb: int) -> List[int]:
        dqbs = self._get_data_qubits()
        print(self.circ)
        neighbors = [qb for qb in self.circ.get_neighbour_qbs(qb) if qb in dqbs]
        return neighbors

    def get_dqb_coords(self) -> Dict[int, Tuple[float, float]]:
        dqb_coords = {}
        dqbs = self._get_data_qubits()
        for i, coord in self.circ.dqb_coords.items():
            if i in dqbs:
                dqb_coords[i] = coord
        return dqb_coords

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
            circ = [("R", self._get_data_qubits())]
            circ += [("Barrier", [])]
            return circ
        elif state in [1, "1"]:
            circ = [("R", self._get_data_qubits())]
            circ += self.x()
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
        self, split_qbs: List[int], new_ids: Tuple[str, str]
    ) -> Tuple[CircuitList, LogicalQubit, LogicalQubit, str]:
        pass


@dataclass()
class RotSurfCode(LogicalQubit):
    """Class representing one logical qubit on a code patch.
    Defining multiple logical qubit inside the same code patch is not
    supported.
    """

    d: Optional[int] = (
        None  # Code distance along both axes. If d is provided, dx and dz cannot be provided separately
    )
    dx: Optional[int] = None  # Number of rows, minimum length of the X operator
    dz: Optional[int] = None  # Number of columns, minimum length of the Z operator
    # So left and right boundary are Z boundaries
    # Top and bottom boundary are X boundaries

    def __deepcopy__(self, memo):
        new_qb = RotSurfCode(
            id=self.id,
            stabilizers=self.stabilizers,
            log_x=self.log_x,
            log_z=self.log_z,
            exists=self.exists,
            dqb_coords=self.dqb_coords,
            aqb_coords=self.aqb_coords,
            log_x_corrections=self.log_x_corrections,
            log_z_corrections=self.log_z_corrections,
            logical_readouts=self.logical_readouts,
            circ=None,
            d=self.d,
            dx=self.dx,
            dz=self.dz,
        )
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
            anc_idx = (
                self.dx * self.dz
            )  # Index of the first ancilla qubit. The data qubits are indexed from 0 to dx*dz-1 and the ancilla qubits are indexed from dx*dz to dx*dz+num_anc-1

            aqb_coords = {}

            ## ZZZZ stabilizers
            for row in range(self.dx - 1):
                for col in range(self.dz - 1):
                    if (row + col) % 2 == 1:
                        pauli_str = "ZZZZ"
                        stabs.append(
                            Stabilizer(
                                pauli_op=PauliOp(
                                    pauli_string=pauli_str,
                                    data_qubits=[
                                        row * self.dz + col,
                                        row * self.dz + col + 1,
                                        (row + 1) * self.dz + col,
                                        (row + 1) * self.dz + col + 1,
                                    ],
                                ),
                                anc_qubits=[anc_idx],
                            )
                        )
                        aqb_coords[anc_idx] = (row + 1.5, col + 1.5)
                        anc_idx += 1

            ## ZZ stabilizers
            for row in range(self.dx - 1):
                if row % 2 == 0:
                    stabs.append(
                        Stabilizer(
                            pauli_op=PauliOp(
                                pauli_string="ZZ",
                                data_qubits=[
                                    row * self.dz,
                                    (row + 1) * self.dz,
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
                                    row * self.dz + self.dz - 1,
                                    (row + 1) * self.dz + self.dz - 1,
                                ],
                            ),
                            anc_qubits=[anc_idx],
                        )
                    )
                    aqb_coords[anc_idx] = (row + 1.5, self.dz + 0.5)
                    anc_idx += 1

            ## XXXX stabilizers
            for row in range(self.dx - 1):
                for col in range(self.dz - 1):
                    if (row + col) % 2 == 0:
                        pauli_str = "XXXX"
                        stabs.append(
                            Stabilizer(
                                pauli_op=PauliOp(
                                    pauli_string=pauli_str,
                                    data_qubits=[
                                        row * self.dz + col,
                                        row * self.dz + col + 1,
                                        (row + 1) * self.dz + col,
                                        (row + 1) * self.dz + col + 1,
                                    ],
                                ),
                                anc_qubits=[anc_idx],
                            )
                        )
                        aqb_coords[anc_idx] = (row + 1.5, col + 1.5)
                        anc_idx += 1

            ## XX stabilizers
            for col in range(self.dz - 1):
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
                                    (self.dx - 1) * self.dz + col,
                                    (self.dx - 1) * self.dz + col + 1,
                                ],
                            ),
                            anc_qubits=[anc_idx],
                        )
                    )
                    aqb_coords[anc_idx] = (self.dx + 0.5, col + 1.5)
                    anc_idx += 1

            dqb_coords = {}
            for i in range(self.dx * self.dz):
                dqb_coords[i] = (1 + i // self.dz, 1 + i % self.dz)

            self.stabilizers = stabs
            self.dqb_coords = dqb_coords
            self.aqb_coords = aqb_coords
            self.log_x = PauliOp(
                pauli_string="X" * self.dx,
                data_qubits=list(np.arange(self.dx) * self.dz),
            )
            self.log_z = PauliOp(pauli_string="Z" * self.dz, data_qubits=range(self.dz))

    def _get_qb_id_from_coords(self, row: int, col: int) -> int:
        return row * self.dz + col

    def transversal_h(self) -> CircuitList:
        circuit_list = []
        for qbs in self._get_data_qubits():
            circuit_list.append(["H", qbs])
        return circuit_list

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
        completed = False
        while not completed:
            for next_qb in self.get_neighbour_qbs(current_qb):
                if next_qb in log_op_dqbs:
                    continue

                if pauli_charges[next_qb] == "Y":
                    log_op_dqbs.append(next_qb)
                    current_qb = next_qb
                    completed = True
                    break
                elif pauli_charges[next_qb] == basis:
                    log_op_dqbs.append(next_qb)
                    current_qb = next_qb
                    break

        return PauliOp(pauli_string=basis * len(log_op_dqbs), data_qubits=log_op_dqbs)

    def get_def_log_x(self) -> PauliOp:
        return self.get_def_log_op("X")

    def get_def_log_z(self) -> PauliOp:
        return self.get_def_log_op("Z")

    def split(
        self, split_qbs: List[int], new_ids: Tuple[str, str]
    ) -> Tuple[CircuitList, LogicalQubit, LogicalQubit, str]:
        threshold = 1e-3

        def find_split_direction(dqb_coords, split_qbs) -> Tuple[str, float]:
            """Finds the direction of the split (either horizontal or vertical),
            based on the coordinates of the split qubits and the coordinates of the
            data qubits."""
            split_direction_hor = True
            split_direction_vert = True
            for qb in split_qbs:
                if dqb_coords[qb][0] != dqb_coords[split_qbs[0]][0]:
                    split_direction_hor = False
                if dqb_coords[qb][1] != dqb_coords[split_qbs[0]][1]:
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
                split_direction = "horizontal"
                splitting_coord = dqb_coords[split_qbs[0]][0]
            else:
                split_direction = "vertical"
                splitting_coord = dqb_coords[split_qbs[0]][1]

            return split_direction, splitting_coord

        def check_which_log_op_is_split(split_qbs, log_x, log_z) -> str:
            """Returns `X` if the logical X operator is split, `Z` if the logical Z
            operator is split."""
            if len(set(split_qbs).intersection(set(log_x.data_qubits))) == 0:
                x_op_split = False  # The logical x operator is not split
            else:
                x_op_split = True  # The logical x operator is split

            if len(set(split_qbs).intersection(set(log_z.data_qubits))) == 0:
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

        # Find the operator which is split into two parts (either X or Z)
        split_operator = check_which_log_op_is_split(split_qbs, self.log_x, self.log_z)

        # Split up the stabilizers
        split_direction, splitting_coord = find_split_direction(
            self.get_dqb_coords(), split_qbs
        )

        def find_new_stabs(
            coordinate_id: int,
            stabilizers: List[Stabilizer],
            dqb_coords,
            check_condition,
        ):
            new_stabs = []
            for stab in stabilizers:
                num_dqbs_on_new_patch = 0  # Number of data qubits of this stabilizer which are on the left side of the split
                for dqb in stab.pauli_op.data_qubits:
                    if check_condition(dqb_coords[dqb][coordinate_id]):
                        num_dqbs_on_new_patch += 1
                if num_dqbs_on_new_patch == 0:
                    new_stabs.append(stab)
                elif num_dqbs_on_new_patch > 0 and num_dqbs_on_new_patch < len(
                    stab.pauli_op.data_qubits
                ):
                    # The stabilizer lies in the boundary region of the split
                    if set(stab.pauli_op.pauli_string) != set(split_operator):
                        continue
                    else:
                        # This stabilizer has to be modified to exclude the measured
                        # split qubits
                        new_pauli_str = ""
                        new_data_qbs = []
                        for i, dqb in enumerate(stab.pauli_op.data_qubits):
                            if (
                                abs(dqb_coords[dqb][coordinate_id] - splitting_coord)
                                > threshold
                            ):
                                new_pauli_str += stab.pauli_op.pauli_string[i]
                                new_data_qbs.append(dqb)
                        new_stab = Stabilizer(
                            pauli_op=PauliOp(
                                pauli_string=new_pauli_str, data_qubits=new_data_qbs
                            ),
                            anc_qubits=stab.anc_qubits,
                            reset=stab.reset,
                        )
                        new_stabs.append(new_stab)
            return new_stabs

        if split_direction == "horizontal":
            new_stabs_left = find_new_stabs(
                coordinate_id=0,
                stabilizers=self.stabilizers,
                dqb_coords=self.get_dqb_coords(),
                check_condition=lambda x: x > splitting_coord - threshold,
            )
            new_stabs_right = find_new_stabs(
                coordinate_id=0,
                stabilizers=self.stabilizers,
                dqb_coords=self.get_dqb_coords(),
                check_condition=lambda x: x < splitting_coord + threshold,
            )

            def construct_new_log_qb(new_stabs, new_id):
                dqbs_new = set(
                    [dqb for stab in new_stabs for dqb in stab.pauli_op.data_qubits]
                )
                aqbs_new = set([aqb for stab in new_stabs for aqb in stab.anc_qubits])
                dqb_coords_new = {qb: self.get_dqb_coords()[qb] for qb in dqbs_new}
                aqb_coords_new = {qb: self.aqb_coords[qb] for qb in aqbs_new}
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
                    dqb_coords=dqb_coords_new,
                    aqb_coords=aqb_coords_new,
                    circ=self.circ,  # Associate them with the same circuit
                )

                print(split_operator)
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

        else:  # Vertical split
            print("Vertical split!")
            pass

        print("Construct first qb")
        new_qb1 = construct_new_log_qb(new_stabs_left, new_ids[0])
        print("Construct second qb")
        new_qb2 = construct_new_log_qb(new_stabs_right, new_ids[1])
        print("Done")

        split_qbs_mmt_circ = []
        if split_operator == "X":
            split_qbs_mmt_circ += [
                ["H", split_qbs],
            ]
        split_qbs_mmt_circ += [
            ["M", split_qbs],
        ]

        return split_qbs_mmt_circ, new_qb1, new_qb2, split_operator

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
                    f"Cannot remove {num_rows} columns since there are only {self.dz} columns. We need at least one column remaining after shrinking."
                )

            qubits_to_measure = []

        elif direction in ["t", "b", "l"]:
            raise NotImplementedError(
                "Shrinking in this direction is not yet supported."
            )
        else:
            raise ValueError("Invalid direction. Must be in ['t', 'b', 'l', 'r']")

        return circuit_list
