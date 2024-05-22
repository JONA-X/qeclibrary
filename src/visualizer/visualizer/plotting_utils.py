from __future__ import annotations
from typing import Union, List, Dict, Tuple

import numpy as np


def sort_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    center_point = np.mean(points, axis=0)
    points_w_angle = []
    for i, point in enumerate(points):
        diff = point - center_point
        points_w_angle.append([
            point[0],
            point[1],
            np.arctan2(
                diff[1], diff[0]
            )
        ]
        )
    sorted_points = np.array(sorted(points_w_angle, key=lambda x: x[-1]))
    return sorted_points[:,0:-1] # Remove the angle column