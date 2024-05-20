from __future__ import annotations
from typing import Union, List, Dict

import numpy as np

from .logical_qubit import LogicalQubit
from .circuit import Circuit
import stim


def circ_log_QST_results(
    circuit: Circuit,
    log_qbs: Union[None, List[str]] = None,
    num_shots: int = 1000,
    bases: List[str] = ["X", "Y", "Z"],
) -> Dict[str, float]:
    if log_qbs is None:
        log_qbs = [c.id for c in circuit.logical_qubits if c.exists is True]

    results_dict = {}
    for basis, c in circuit.log_QST(log_qbs, bases):
        res = stim.Circuit(c.convert_to_stim()).compile_sampler().sample(num_shots)
        summed_res = 0
        for r in res:
            shot_readout = 0
            mmts = c.dict_m_labels_to_res(r)
            for qb in log_qbs:
                final_readout = np.array(mmts[f"QST_{qb}"], dtype=int)
                shot_readout += np.sum(final_readout)
            summed_res += shot_readout % 2
        summed_res /= num_shots
        results_dict[basis] = summed_res
    return results_dict
