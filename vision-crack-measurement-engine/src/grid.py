"""Grid generation utilities."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


LineSegment = Tuple[Tuple[int, int], Tuple[int, int]]


def generate_perspective_grid(
    inverse_homography: np.ndarray,
    feet_per_pixel_x: float,
    feet_per_pixel_y: float,
    lane_polygon: np.ndarray,
    lane_length_ft: int = 35,
    lane_width_ft: int = 11,
) -> List[LineSegment]:
    """Generate 1x1 ft grid lines and warp them back to image perspective.

    Grid lines are retained only when both warped endpoints lie inside the lane polygon.

    Args:
        inverse_homography: Transform from top-down plane to image plane.
        feet_per_pixel_x: Top-down x-axis scale in feet/pixel.
        feet_per_pixel_y: Top-down y-axis scale in feet/pixel.
        lane_polygon: Lane polygon in image coordinates, shape ``(N, 2)``.
        lane_length_ft: Top-down lane length in feet.
        lane_width_ft: Top-down lane width in feet.

    Returns:
        List of clipped-valid line segments as ``[((x1, y1), (x2, y2)), ...]``.
    """
    if feet_per_pixel_x <= 0 or feet_per_pixel_y <= 0:
        raise ValueError("feet_per_pixel values must be positive.")
    if lane_polygon.ndim != 2 or lane_polygon.shape[1] != 2:
        raise ValueError(f"lane_polygon must be (N, 2), got {lane_polygon.shape}")

    max_x_px = lane_length_ft / feet_per_pixel_x
    max_y_px = lane_width_ft / feet_per_pixel_y

    segments_topdown: list[tuple[np.ndarray, np.ndarray]] = []

    for ft_x in range(lane_length_ft + 1):
        x_px = ft_x / feet_per_pixel_x
        p1 = np.array([x_px, 0.0], dtype=np.float32)
        p2 = np.array([x_px, max_y_px], dtype=np.float32)
        segments_topdown.append((p1, p2))

    for ft_y in range(lane_width_ft + 1):
        y_px = ft_y / feet_per_pixel_y
        p1 = np.array([0.0, y_px], dtype=np.float32)
        p2 = np.array([max_x_px, y_px], dtype=np.float32)
        segments_topdown.append((p1, p2))

    lane_contour = lane_polygon.astype(np.float32).reshape(-1, 1, 2)

    output: List[LineSegment] = []
    for p1, p2 in segments_topdown:
        pts = np.array([p1, p2], dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, inverse_homography).reshape(-1, 2)

        inside_a = cv2.pointPolygonTest(lane_contour, (float(warped[0, 0]), float(warped[0, 1])), False) >= 0
        inside_b = cv2.pointPolygonTest(lane_contour, (float(warped[1, 0]), float(warped[1, 1])), False) >= 0
        if not (inside_a and inside_b):
            continue

        a = (int(round(warped[0, 0])), int(round(warped[0, 1])))
        b = (int(round(warped[1, 0])), int(round(warped[1, 1])))
        output.append((a, b))

    return output
