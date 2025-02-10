from abc import ABC, abstractmethod
from .definitions import CircuitList


class NoiseModel(ABC):
    @abstractmethod
    def add_errors_to_circuit(
        self,
        op_list: CircuitList,
    ) -> CircuitList:
        pass


class PauliNoiseModel(NoiseModel):
    def __init__(
        self,
        p: float,
        p_2q: float = None,
        p_reset: float = None,
        p_mmt: float = None,
    ):
        self.p = p

        if p_2q is not None:
            self.p_2q = p_2q
        else:
            self.p_2q = p

        if p_reset is not None:
            self.p_reset = p_reset
        else:
            self.p_reset = p

        if p_mmt is not None:
            self.p_mmt = p_mmt
        else:
            self.p_mmt = p

    def add_errors_to_circuit(
        self,
        op_list: CircuitList,
    ) -> CircuitList:
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
            elif op[0][:18] == "OBSERVABLE_INCLUDE":
                # Don't add any errors to the observable definitions
                op_list_with_errors += [
                    (op[0], op[1]),
                ]
            else:
                op_list_with_errors += [
                    (op[0], op[1]),
                    ("DEPOLARIZE1", op[1], self.p),
                ]

        return op_list_with_errors
