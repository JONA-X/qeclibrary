import stim
import importlib
import qeclib
from qeclib import convert_to_stim
from qeclib import PauliNoiseModel

def get_circ(stabilizers, n, n_qec_rounds, init_state, measure_op, qb_coords, incl_det_ids, stab_ext_mode):
    measure_op_F2 = stab_to_F2([[measure_op, range(n), None]], n)

    circ = []
    if init_state == "ZZ":
        circ += []
    elif init_state == "XX":
        circ += [
            ("H", range(n)),
        ]
    else:
        raise ValueError("Invalid initial state")
    circ += [("Barrier", [])]
    circ += get_syndrome_circuit(stabilizers, n_qec_rounds, n_qubits=n, prep_basis=measure_op, qb_coords=qb_coords, incl_det_ids=incl_det_ids, mode=stab_ext_mode)

    if "X" in measure_op:
        circ += [("MX", [i for i in range(len(measure_op)) if measure_op[i] == "X"])]
    if "Y" in measure_op:
        circ += [("MY", [i for i in range(len(measure_op)) if measure_op[i] == "Y"])]
    if "Z" in measure_op:
        circ += [("M", [i for i in range(len(measure_op)) if measure_op[i] == "Z"])]

    for stab_index, stab in enumerate(stabilizers):
        if sim_readout(stab_to_F2([stab], n)[0], measure_op_F2[0]) and stab_index in incl_det_ids:
            circ += [("DETECTOR", [f"rec[{-n+stab[1][i]}]" for i in range(len(stab[1])) if stab[0][i] != "I"] + [f"rec[{-n-len(stabilizers) + stab_index}]"], f"{qb_coords[stab[2]][0]}, {qb_coords[stab[2]][1]}, {n_qec_rounds}")]

    circ += [("OBSERVABLE_INCLUDE", [f"rec[{-n+i}]" for i in range(len(log_op)) if log_op[i] != "I"], 0)]

    return circ

def get_stim_circ(stabilizers, n, n_qec_rounds, init_state, measure_op, qb_coords, incl_det_ids, stab_ext_mode):
    circ = get_circ(stabilizers, n, n_qec_rounds, init_state, measure_op, qb_coords, incl_det_ids, stab_ext_mode)
    noise_model = PauliNoiseModel(0.01)
    stim_circ = qeclib.convert_to_stim(circ, qb_coords, noise_model)
    return stim.Circuit(stim_circ)


def get_syndrome_circuit(stabilizers, n_cycles: int = 1, n_qubits: int = 0, prep_basis: str = "", qb_coords: dict = None, incl_det_ids: list[int] = None, mode: str = "parallel"):
    if incl_det_ids is None:
        incl_det_ids = list(range(len(stabilizers)))
    def get_qb_coord(i):
        if qb_coords is None:
            return i
        else:
            return f"{qb_coords[i][0]}, {qb_coords[i][1]}"
    ancilla_qbs = [stab[2] for stab in stabilizers]
    max_stab_length = 0
    for stab in stabilizers:
        max_stab_length = max(max_stab_length, len(stab[0]))

    circuit_list = []
    for qec_cycle in range(n_cycles):
        circuit_list += (
            ("H", [id for id in ancilla_qbs]),
            ("Barrier", []),
        )

        if mode == "sequential":
            for stab in stabilizers:
                for step in range(max_stab_length):
                    if stab[0][step] != "I":
                        circuit_list += (
                            (
                                f"C{stab[0][step]}",
                                [
                                    stab[2],
                                    stab[1][step],
                                ],
                            ),
                        )
                circuit_list += (
                    ("H", stab[2]),
                    ("MR", stab[2]),
                    ("Barrier", []),
                )
        elif mode == "parallel":
            for step in range(max_stab_length):
                for stab in stabilizers:
                    if stab[0][step] != "I":
                        circuit_list += (
                            (
                                f"C{stab[0][step]}",
                                [
                                    stab[2],
                                    stab[1][step],
                                ],
                            ),
                        )
                circuit_list += (("Barrier", []),)

            circuit_list += (
                ("H", ancilla_qbs),
                ("MR", ancilla_qbs),
            )

        # Create detectors
        if qec_cycle >= 1:
            circuit_list += [
                ("DETECTOR", [f"rec[{-len(stabilizers)*2 + i}]", f"rec[{-len(stabilizers) + i}]"], f"{get_qb_coord(stabilizers[i][2])}, {qec_cycle}")
                for i in range(len(stabilizers))
                if i in incl_det_ids
            ]
        if qec_cycle == 0:
            circuit_list += [
                ("DETECTOR", [f"rec[{-len(stabilizers) + i}]"], f"{get_qb_coord(stabilizers[i][2])}, {qec_cycle}")
                for i in range(len(stabilizers))
                if prep_basis != "" and sim_readout(stab_to_F2([stabilizers[i]], n_qubits)[0], stab_to_F2([[prep_basis, range(n_qubits), None]], n_qubits)[0])
                and i in incl_det_ids
            ]
        circuit_list += (
            ("Barrier", []),
        )

    return circuit_list

