import numpy as np
from ..definitions import StabilizerTuple


def stab_to_F2(stabilizer_list: list[StabilizerTuple], n: int = None) -> np.ndarray:
    """Convert a list of stabilizers in the form `(pauli_string, [data_qubit_indices],
    ancilla_index)` to a matrix where every row corresponds to a stabilizer in binary
    symplectic vector representation. The returned stabilizer matrix ignores ancilla
    qubits.

    Parameters
    ----------
    stabilizer_list : list[StabilizerTuple]
        List of stabilizers where every stabilizer is given as a tuple
        `(pauli_string, [data_qubit_indices], ancilla_index)` where `pauli_string`
        contains X, Y, Z, and I, `[data_qubit_indices]` is a list of the involved data
        qubit indices.
    n : int
        Number of qubits in the system.

    Returns
    -------
    np.ndarray
        Matrix representation of the stabilizers in binary symplectic vector form.
    """
    # If the number of qubits is not provided, assume that it is the maximum qubit index
    # in the stabilizer list plus one
    if n is None:
        n = (
            np.max(
                [
                    np.max(
                        [
                            stab[1][i]
                            for i in range(len(stab[1]))
                            if stab[1][i] is not None
                        ]
                    )
                    for stab in stabilizer_list
                ]
            )
            + 1
        )

    stab_matrix = []
    for stab in stabilizer_list:
        stab_F2 = np.zeros(2 * n, dtype=int)
        for i, x in enumerate(stab[0]):
            if x in ["X", "Y"]:
                stab_F2[stab[1][i]] += 1
            if x in ["Z", "Y"]:
                stab_F2[stab[1][i] + n] += 1
        stab_matrix.append(stab_F2)
    return np.array(stab_matrix)
