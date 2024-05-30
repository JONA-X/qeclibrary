from pydantic import Field
from pydantic.dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qeclib import Circuit, Stabilizer
from .plotting_utils import sort_points


@dataclass()
class QECPlot:
    circ: Circuit = None
    show_grid: bool = False
    x_axis_visible: bool = False
    y_axis_visible: bool = False
    width: int = 1100
    height: int = 700
    _log_qb_counter: int = (
        0  # Counter for logical qubits to automatically choose different colors for them
    )
    _log_qb_default_color: tuple[str] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    _colors_XYZ: dict[str, str] = Field(
        default_factory=lambda: {"X": "#5AAEE6", "Y": "#F0A0E9", "Z": "#57C95A"}
    )

    def __post_init__(self) -> None:
        # Create the figure object
        self._fig = make_subplots(rows=1, cols=1)

        # Default plot settings
        self._fig.update_layout(
            width=self.width,
            height=self.height,
            showlegend=True,
            xaxis={
                "showgrid": self.show_grid,
                "zeroline": False,
                "visible": self.x_axis_visible,
            },
            yaxis={
                "showgrid": self.show_grid,
                "visible": self.y_axis_visible,
                "scaleanchor": "x",
                "scaleratio": 1,
                "autorange": True,  # Otherwise y axis would be reversed
            },
        )

        # Plot data and ancilla qubits
        if self.circ is not None:
            self.add_dqubits(
                self.circ.dqb_coords,
                color="gray",
                number_inside_marker=False,
                name="Data qubits",
                showlegend=True,
                legendgroup="qpu_dqbs",
            )
            self.add_dqubits(
                self.circ.aqb_coords,
                color="lightgray",
                number_inside_marker=False,
                marker_size=15,
                name="Ancilla qubits",
                showlegend=True,
                legendgroup="qpu_aqbs",
            )

    def show(self):
        self._fig.show()

    def add_dqubits(
        self,
        dqb_coords,
        color: str = "gray",
        number_inside_marker: bool = True,
        marker_size: int = 25,
        name: str = "",
        showlegend: bool = False,
        legendgroup: str = "",
        marker_symbol: str = "circle",
        text_dict: dict = None,
    ):
        if number_inside_marker:
            mode = "markers+text"
        else:
            mode = "markers"

        i = 0
        for qb_id, qb in dqb_coords.items():
            self._fig.add_trace(
                go.Scatter(
                    x=[qb[1]],
                    y=[qb[0]],
                    mode=mode,
                    name=name,
                    marker=dict(
                        size=marker_size,
                        color=color,
                        line=dict(
                            width=1,
                            color="darkred",
                        ),
                        symbol=marker_symbol,
                    ),
                    text=qb_id if text_dict is None else text_dict[qb_id],
                    textposition="middle center",
                    hoverinfo="text",
                    legendgroup=legendgroup,
                    showlegend=(showlegend and i == 0),
                )
            )
            i += 1

    def plot_pauli_string(self, pauli_string: str, dqubits: list[int]):
        dqb_coords_pauli = {}
        text_dict = {}
        for pauli, qb in zip(pauli_string, dqubits):
            dqb_coords_pauli[qb] = self.circ.dqb_coords[qb]
            text_dict[qb] = pauli + "<sub>" + str(qb) + "</sub>"
        self.add_dqubits(
            dqb_coords_pauli,
            marker_symbol="square",
            text_dict=text_dict,
        )

    def plot_stabilizers(self, obj, legend_qb: str = "", name: str = ""):
        if isinstance(obj, str):
            self.circ.log_qb_id_valid_check(obj)
            stabs = self.circ.log_qbs[obj].stabilizers
            if legend_qb == "":
                legend_qb = self.circ.log_qbs[obj].id
            if name == "":
                name = f"{self.circ.log_qbs[obj].id} stabilizers"
        elif isinstance(obj, list) and not any(
            not isinstance(stab, Stabilizer) for stab in obj
        ):
            stabs = obj
        else:
            raise ValueError("Invalid input for plot_stabilizers")

        for i, stab in enumerate(stabs):
            is_css = True if len(set(stab.pauli_op.pauli_string)) == 1 else False
            if is_css:
                color = self._colors_XYZ[stab.pauli_op.pauli_string[0]]
            else:
                color = self._colors_XYZ["Z"]

            coords = [self.circ.dqb_coords[qb] for qb in stab.pauli_op.data_qubits]
            s_coords = list(sort_points(coords))
            s_coords.append(s_coords[-1])
            s_coords = np.array(s_coords)
            self._fig.add_trace(
                go.Scatter(
                    x=s_coords[:, 1],
                    y=s_coords[:, 0],
                    mode="lines",
                    name=name,
                    fill="toself",
                    fillcolor=color,
                    line=dict(
                        color="black",
                        width=2,
                    ),
                    hoverinfo="none",
                    legendgroup=f"stabs_{legend_qb}",
                    showlegend=i == 0,
                )
            )

    def plot_logical_qubit(self, qb_id: str):
        self.circ.log_qb_id_valid_check(
            qb_id
        )  # Check logical qubit id and throw error if invalid
        self.plot_stabilizers(qb_id)  # Plot stabilizers
        self.add_dqubits(
            self.circ.log_qbs[qb_id].get_dqb_coords(),
            color=self._log_qb_default_color[self._log_qb_counter],
        )  # Plot data qubits
        self.plot_pauli_string(
            self.circ.log_qbs[qb_id].log_x.pauli_string,
            self.circ.log_qbs[qb_id].log_x.data_qubits,
        )
        self.plot_pauli_string(
            self.circ.log_qbs[qb_id].log_z.pauli_string,
            self.circ.log_qbs[qb_id].log_z.data_qubits,
        )

        self._log_qb_counter += 1
