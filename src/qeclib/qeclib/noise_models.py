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
            # Reset
            if op[0] == "R":
                op_list_with_errors += [
                    (op[0], op[1]),
                ]
                if self.p_reset > 0:
                    op_list_with_errors += [
                        ("DEPOLARIZE1", op[1], self.p_reset),
                    ]
            # Measurement
            elif op[0] in ["M", "MX", "MY"]:
                if self.p_mmt > 0:
                    op_list_with_errors += [
                        ("DEPOLARIZE1", op[1], self.p_mmt),
                    ]
                op_list_with_errors += [
                    (op[0], op[1]),
                ]
            # Measurement and reset
            elif op[0] == "MR":
                if self.p_mmt > 0:
                    op_list_with_errors += [
                        ("DEPOLARIZE1", op[1], self.p_mmt),
                    ]
                op_list_with_errors += [
                    (op[0], op[1]),
                ]
                if self.p_reset > 0:
                    op_list_with_errors += [
                        ("DEPOLARIZE1", op[1], self.p_reset),
                    ]
            # Controlled gates
            elif op[0] in ["CX", "CY", "CZ"]:
                op_list_with_errors += [
                    (op[0], op[1]),
                ]
                if self.p_2q > 0:
                    op_list_with_errors += [
                        ("DEPOLARIZE2", op[1], self.p_2q),
                    ]
            # Observables
            elif op[0][:18] == "OBSERVABLE_INCLUDE" or op[0] == "DETECTOR":
                op_list_with_errors += [
                    tuple([op[i] for i in range(len(op))]),
                ]
            # All other operations
            else:
                op_list_with_errors += [
                    (op[0], op[1]),
                ]
                if self.p > 0:
                    op_list_with_errors += [
                        ("DEPOLARIZE1", op[1], self.p),
                    ]

        return op_list_with_errors
