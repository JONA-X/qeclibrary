from .stabilizer import Stabilizer
from .noise_models import NoiseModel
from .definitions import internal_op_to_stim_map, CircuitList


def number_of_data_qubits_in_stab_list(stabs: list[Stabilizer]) -> int:
    """Returns the number of data qubits involved in the given list of
    stabilizers.

    Returns:
        int: Number of data qubits in the given list of stabilizers
    """
    data_qubit_indices = []
    for stab in stabs:
        for qb in stab.data_qubits:
            if qb not in data_qubit_indices:
                data_qubit_indices.append(qb)
    return len(data_qubit_indices)


def pauli_product(pauli1: str, pauli2: str) -> str:
    if len(pauli1) != len(pauli2):
        raise ValueError("Pauli strings must have the same length")

    product = ""
    sign = 1
    for i in range(len(pauli1)):
        if pauli1[i] == "I":
            product += pauli2[i]
        elif pauli2[i] == "I":
            product += pauli1[i]
        elif pauli1[i] == pauli2[i]:
            product += "I"
        else:
            if pauli1[i] == "X":
                if pauli2[i] == "Y":
                    product += "Z"
                    sign *= 1j
                else:
                    product += "Y"
                    sign *= -1j
            elif pauli1[i] == "Y":
                if pauli2[i] == "X":
                    product += "Z"
                    sign *= -1j
                else:
                    product += "X"
                    sign *= 1j
            else:
                if pauli2[i] == "X":
                    product += "Y"
                    sign *= 1j
                else:
                    product += "X"
                    sign *= -1j
    return product, sign


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
        stim_circ += internal_op_to_stim_map[op[0]]
        if op[0] in ["DEPOLARIZE1", "DEPOLARIZE2"]:
            stim_circ += f"({op[2]})"
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
