"""Longitudinal crack length estimation."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def compute_total_crack_length_ft(
    longitudinal_polygons: List[np.ndarray],
    homography: np.ndarray,
    feet_per_pixel_x: float,
    feet_per_pixel_y: float,
) -> float:
    """Compute total longitudinal crack length in feet.

    Args:
        longitudinal_polygons: Crack polygons in image pixel coordinates.
        homography: Perspective transform from image space to top-down space.
        feet_per_pixel_x: X-axis scale in feet/pixel in top-down plane.
        feet_per_pixel_y: Y-axis scale in feet/pixel in top-down plane.

    Returns:
        Total crack length in feet.
    """
    total_length_ft = 0.0

    for polygon in longitudinal_polygons:
        if len(polygon) < 2:
            continue

        pts = polygon.astype(np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)

        if len(warped) >= 4:
            # For region-like longitudinal polygons, use the dominant axis length.
            rect = cv2.minAreaRect(warped.astype(np.float32))
            box = cv2.boxPoints(rect).astype(np.float32)
            edges = []
            for i in range(4):
                p1 = box[i]
                p2 = box[(i + 1) % 4]
                dx_ft = (p2[0] - p1[0]) * feet_per_pixel_x
                dy_ft = (p2[1] - p1[1]) * feet_per_pixel_y
                edges.append(float(np.hypot(dx_ft, dy_ft)))
            total_length_ft += max(edges)
            continue

        deltas = np.diff(warped, axis=0)
        segment_lengths_ft = np.sqrt(
            (deltas[:, 0] * feet_per_pixel_x) ** 2
            + (deltas[:, 1] * feet_per_pixel_y) ** 2
        )
        total_length_ft += float(segment_lengths_ft.sum())

    return total_length_ft
