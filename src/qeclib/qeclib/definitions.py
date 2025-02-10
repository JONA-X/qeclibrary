CircuitList = list[tuple[str, list[int | tuple[int, int]]]]

internal_op_to_stim_map: dict[str, str] = {
    "R": "R",
    "RX": "RX",
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
