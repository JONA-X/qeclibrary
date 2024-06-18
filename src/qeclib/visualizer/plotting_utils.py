import numpy as np


def sort_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    center_point = np.mean(points, axis=0)
    points_w_angle = []
    for i, point in enumerate(points):
        diff = point - center_point
        points_w_angle.append([point[0], point[1], np.arctan2(diff[1], diff[0])])
    sorted_points = np.array(sorted(points_w_angle, key=lambda x: x[-1]))
    return sorted_points[:, 0:-1]  # Remove the angle column

def hex_to_rgb(hex_color: str) -> list[int]:
    if hex_color[0] == "#":
        hex_color = hex_color[1:]

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return [r, g, b]
