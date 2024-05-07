from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import stim

from .logical_qubit import LogicalQubit


def circ_log_QST_results(
    circuit, log_qbs: Union[None, List[LogicalQubit]] = None, num_shots: int = 1000
) -> Dict[str, float]:
    if log_qbs == None:
        log_qbs = circuit.logical_qubits

    results_dict = {}
    for basis, c in circuit.log_QST(log_qbs):
        res = stim.Circuit(c.convert_to_stim()).compile_sampler().sample(num_shots)
        summed_res = 0
        for r in res:
            mmts = c.dict_m_labels_to_res(r)
            for qb in log_qbs:
                final_readout = np.array(mmts[f"QST_{qb.id}"], dtype=int)
                summed_res += np.sum(final_readout) % 2
        summed_res /= num_shots
        results_dict[basis] = summed_res
    return results_dict
