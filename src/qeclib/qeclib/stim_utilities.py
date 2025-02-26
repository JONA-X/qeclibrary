import stim
import pymatching
import numpy as np
from .noise_models import NoiseModel, PauliNoiseModel
from .definitions import internal_op_to_stim_map, CircuitList


def convert_to_stim(
    circuit: CircuitList,
    qb_coords: dict[str | int, tuple[float, float]],
    noise_model: NoiseModel = None,
) -> str:
    stim_circ = ""
    for qb_id, coords in qb_coords.items():
        stim_circ += f"QUBIT_COORDS({coords[0]},{coords[1]}) {qb_id}\n"

    if noise_model is None:
        operation_list = circuit
    else:
        operation_list = noise_model.add_errors_to_circuit(circuit)

    # Operations of the circuit
    for op in operation_list:
        # Operation name
        stim_circ += internal_op_to_stim_map[op[0]]

        # Additional parameters
        if op[0] in ["DEPOLARIZE1", "DEPOLARIZE2", "OBSERVABLE_INCLUDE", "DETECTOR"]:
            stim_circ += f"({op[2]})"

        # Qubits/measurements that are involved
        if isinstance(op[1], int):
            stim_circ += f" {op[1]}"
        else:
            for qb in op[1]:
                if isinstance(qb, list):
                    raise NotImplementedError(
                        f"Referencing qubits by coordinates is not yet implemented. Provided was {qb}"
                    )
                else:
                    stim_circ += f" {qb}"
        stim_circ += "\n"
    return stim_circ
