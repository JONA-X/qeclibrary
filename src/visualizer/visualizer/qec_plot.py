from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple, Literal
from pydantic.dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qeclib import Circuit, Stabilizer, LogicalQubit
from .plotting_utils import sort_points


@dataclass()
class QECPlot:
    circ: Circuit = None
    show_grid: bool = False
    x_axis_visible: bool = False
    y_axis_visible: bool = False
    width: int = 1100
    height: int = 700

    def __post_init__(self) -> None:
        # Create the figure object
        self.fig = make_subplots(rows=1, cols=1)

        # Default plot settings
        self.fig.update_layout(
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
                self.circ.anc_coords,
                color="lightgray",
                number_inside_marker=False,
                marker_size=15,
                name="Ancilla qubits",
                showlegend=True,
                legendgroup="qpu_aqbs",
                )

    def add_dqubits(
            self,
            dqb_coords,
            color: str = "gray",
            number_inside_marker: bool = True,
            marker_size: int = 25,
            name: str = "",
            showlegend: bool = False,
            legendgroup: str = "",
            ):
        if number_inside_marker:
            mode = "markers+text"
        else:
            mode = "markers"

        i = 0
        for qb_id, qb in dqb_coords.items():
            self.fig.add_trace(
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
                            ),
                    text=qb_id,
                    textposition="middle center",
                    hoverinfo="text",
                    legendgroup=legendgroup,
                    showlegend=(showlegend and i==0),
                )
            )
            i += 1

    def plot_stabilizers(self, obj):
        if isinstance(obj, LogicalQubit):
            stabs = obj.stabilizers
            legend_qb = obj.id
        elif isinstance(obj, list) and not any(isinstance(stab, Stabilizer) for stab in obj):
            stabs = obj
            legend_qb = ""
        else:
            raise ValueError("Invalid input for plot_stabilizers")

        for i, stab in enumerate(stabs):
            coords = [self.circ.dqb_coords[qb] for qb in stab.pauli_op.data_qubits]
            s_coords = list(sort_points(coords))
            s_coords.append(s_coords[-1])
            s_coords = np.array(s_coords)
            self.fig.add_trace(
                go.Scatter(
                    x=s_coords[:,1],
                    y=s_coords[:,0],
                    mode="lines",
                    fill="toself",
                    line=dict(
                        color="black",
                        width=2,
                    ),
                    hoverinfo="none",
                    legendgroup=f"stabs_{legend_qb}",
                    showlegend=i==0,
                )
            )