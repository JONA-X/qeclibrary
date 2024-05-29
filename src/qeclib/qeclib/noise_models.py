from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple


CircuitList = List[Tuple[str, List[Union[int, Tuple[int, int]]]]]


class NoiseModel:
    pass


class PauliNoiseModel(NoiseModel):
    def __init__(
        self,
        p: float,
        p_2q: float,
        p_reset: float,
        p_mmt: float,
    ):
        self.p = p
        self.p_2q = p_2q
        self.p_reset = p_reset
        self.p_mmt = p_mmt

    def add_errors_to_circuit(
        self,
        op_list: CircuitList,
    ) -> Dict[str, float]:
        op_list_with_errors = []
        for op in op_list:
            if op[0] == "R":
                op_list_with_errors += [
                    (op[0], op[1]),
                    ("DEPOLARIZE1", op[1], self.p_reset),
                ]
            elif op[0] == "M":
                op_list_with_errors += [
                    ("DEPOLARIZE1", op[1], self.p_mmt),
                    (op[0], op[1]),
                ]
            elif op[0] == "MR":
                op_list_with_errors += [
                    ("DEPOLARIZE1", op[1], self.p_mmt),
                    (op[0], op[1]),
                    ("DEPOLARIZE1", op[1], self.p_reset),
                ]
            elif op[0] in ["CX", "CY", "CZ"]:
                op_list_with_errors += [
                    (op[0], op[1]),
                    ("DEPOLARIZE2", op[1], self.p_mmt),
                ]
            else:
                op_list_with_errors += [
                    (op[0], op[1]),
                    ("DEPOLARIZE1", op[1], self.p),
                ]

        return op_list_with_errors
