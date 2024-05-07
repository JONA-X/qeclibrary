from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple

from .stabilizer import *


def number_of_data_qubits_in_stab_list(stabs: List[Stabilizer]) -> int:
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


def check_commutation_of_pauli_string(str1, str2) -> bool:
    if len(str1) != len(str2):
        raise ValueError("Pauli strings must have the same length")
    do_commute = True
    for i in range(len(str1)):
        if str1[i] != "I" and str2[i] != "I":
            if str1[i] != str2[i]:
                do_commute = not (do_commute)
    return do_commute


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
