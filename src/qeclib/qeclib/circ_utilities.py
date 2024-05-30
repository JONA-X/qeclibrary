import numpy as np

from .logical_qubit import LogicalQubit
from .circuit import Circuit
import stim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def circ_log_QST_results(
    circuit: Circuit,
    log_qbs: list[str] | None = None,
    num_shots: int = 1000,
    bases: list[str] = ["X", "Y", "Z"],
) -> dict[str, float]:
    if log_qbs is None:
        log_qbs = [qb_id for qb_id, qb in circuit.log_qbs.items() if qb.exists is True]

    results_dict = {}
    for basis, c in circuit.log_QST(log_qbs, bases):
        res = stim.Circuit(c.convert_to_stim()).compile_sampler().sample(num_shots)
        labels_to_uuids = c.dict_m_labels_to_uuids()
        summed_res = 0.0
        for r in res:
            shot_readout = 0
            for i, qb in enumerate(log_qbs):
                if basis[i] == "I":
                    continue
                shot_readout += c.get_log_dqb_readout(
                    r, labels_to_uuids[f"QST_{qb}_{basis[i]}"], qb
                )
            summed_res += shot_readout % 2
        summed_res /= num_shots
        results_dict[basis] = summed_res
    return results_dict


def plot_log_QST_results(res_dict, qb_names=["Q1", "Q2"]):
    single_qb_bases = []
    for two_qb_basis in res_dict.keys():
        if two_qb_basis[0] not in single_qb_bases:
            single_qb_bases.append(two_qb_basis[0])

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111, projection="3d")

    x = []
    y = []
    z = []
    for basis, exp_val in res_dict.items():
        x.append(single_qb_bases.index(basis[0]))
        y.append(single_qb_bases.index(basis[1]))
        z.append(exp_val)

    num_elements = len(x)
    bottom = np.zeros(num_elements)
    width = np.ones(num_elements)
    depth = np.ones(num_elements)

    ax1.bar3d(
        x,
        y,
        bottom,
        width,
        depth,
        z,
        color="hotpink",
        alpha=0.5,
        edgecolor=(0.2, 0.2, 0.2, 0.9),
        linewidth=1,
        linestyle="solid",
    )

    xticks = np.arange(0.5, 0.5 + len(single_qb_bases), 1)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(single_qb_bases)
    ax1.set_yticks(xticks)
    ax1.set_yticklabels(single_qb_bases)

    ax1.invert_xaxis()
    plt.xlabel(f"Basis of {qb_names[0]}")
    plt.ylabel(f"Basis of {qb_names[1]}")
    return fig
