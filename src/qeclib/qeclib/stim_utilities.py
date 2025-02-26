import stim
import pymatching
import numpy as np
from .noise_models import NoiseModel, PauliNoiseModel
from .definitions import internal_op_to_stim_map, CircuitList


def convert_to_stim(
    circuit: CircuitList,
    qb_coords: dict[str | int, tuple[float, float]] = None,
    noise_model: NoiseModel = None,
) -> str:
    stim_circ = ""
    # If provided, add qubit coordinates
    if qb_coords is not None:
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


def get_logical_error_rate(
    circuit: CircuitList,
    num_shots=1000,
    p: float = None,
    noise_model: NoiseModel = None,
) -> tuple[float, float, float, float]:
    """Get the raw, decoded, and post-selected logical error rates for a given circuit
    and a given noise model or physical error rate.

    Parameters
    ----------
    circuit : CircuitList
        Circuit to be simulated
    num_shots : int, optional
        Number of shots, by default 1000
    p : float, optional
        Physical error rate for a Pauli noise model
    noise_model : NoiseModel, optional
        Noise model for converting the ideal circuit into a noisy one

    Returns
    -------
    tuple[float, float, float, float]
        Raw logical error rate, decoded logical error rate, post-selected logical error
        rate, ratio of logical flips predicted by the decoder
    """
    if noise_model is None and p is None:
        raise ValueError("Either noise_model or p must be provided.")
    if noise_model is not None and p is not None:
        raise ValueError("Only noise_model or p must be provided but not both.")

    if noise_model is None:
        noise_model = PauliNoiseModel(p)

    stim_circuit = stim.Circuit(convert_to_stim(circuit, noise_model=noise_model))
    sampler = stim_circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )
    detector_error_model = stim_circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    predictions = matcher.decode_batch(detection_events)

    # Count the number of flips and errors
    num_raw_errors = 0
    num_dec_flips = 0
    num_dec_errors = 0
    num_pst_errors = 0
    for shot in range(num_shots):
        # If one of the logical observables was flipped: Raw error
        if any(observable_flips[shot]) == 1:
            num_raw_errors += 1

        # If the decoder predicted a logical flip (maybe correctly, maybe not)
        if any(predictions[shot]):
            num_dec_flips += 1

        # If the prediction does not match the actual: Decoding error
        if not np.array_equal(observable_flips[shot], predictions[shot]):
            num_dec_errors += 1

        # If there was a logical error but no detection event: Post-selection error
        if any(observable_flips[shot]) == 1 and np.sum(detection_events[shot]) == 0:
            num_pst_errors += 1

    return (
        num_raw_errors / num_shots,
        num_dec_errors / num_shots,
        num_pst_errors / num_shots,
        num_dec_flips / num_shots,
    )


def print_decoder_performance(
    circuit: CircuitList, p: float, num_shots: int = 100000
) -> None:
    """Print some statistics on how the pymatching decoder performs for the given
    circuit and physical error rate.

    Parameters
    ----------
    circuit : CircuitList
        Circuit to be simulated
    p : float, optional
        Physical error rate for a Pauli noise model
    num_shots : int, optional
        Number of shots, by default 100000

    Returns
    -------
    """
    raw_err_rate, dec_err_rate, pst_err_rate, dec_flip_rate = get_logical_error_rate(
        circuit,
        num_shots=num_shots,
        p=p,
    )
    print(f"Out of {num_shots} shots, there were")
    print(
        f" • {int(raw_err_rate * num_shots)} logical flips ({np.round(100*raw_err_rate, 2)} % of shots),"
    )
    print(
        f" • {int(dec_flip_rate * num_shots)} logical flips predicted by the decoder ({np.round(100*dec_flip_rate, 2)} % of shots), and"
    )
    print(
        f" • {int(dec_err_rate * num_shots)} decoder mistakes ({np.round(100*dec_err_rate, 2)} % of shots)"
    )
    print(
        f" • {int(pst_err_rate * num_shots)} post-selection mistakes ({np.round(100*pst_err_rate, 2)} % of shots)"
    )
