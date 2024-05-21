from __future__ import annotations
from typing import Union, List, Optional, Dict, Tuple, Literal
from pydantic.dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass()
class QECPlot:
    name: str
    show_grid: bool = False
    x_axis_visible: bool = False
    y_axis_visible: bool = False

    def __post_init__(self) -> None:
        # Create the figure object
        self.fig = make_subplots(rows=1, cols=1)

        # Default plot settings
        self.fig.update_layout(
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
