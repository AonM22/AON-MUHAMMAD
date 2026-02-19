"""Lane scaling and projective geometry helpers."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


DEFAULT_LANE_LENGTH_FT = 15.0
DEFAULT_LANE_WIDTH_FT = 12.0


LaneScalingResult = Tuple[np.ndarray, np.ndarray, float, float]


def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
    """Order 4 rectangle points as top-left, top-right, bottom-right, bottom-left."""
    if points.shape != (4, 2):
        raise ValueError(f"Expected 4x2 points, got {points.shape}")

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def compute_lane_scaling(
    lane_polygon: np.ndarray,
    lane_length_ft: float = DEFAULT_LANE_LENGTH_FT,
    lane_width_ft: float = DEFAULT_LANE_WIDTH_FT,
) -> LaneScalingResult:
    """Compute lane homography and per-axis scale from lane polygon.

    Geometric reasoning:
    1. ``cv2.minAreaRect`` estimates the best-fit oriented rectangle for lane points.
    2. The rectangle has two pixel side lengths. Either side may represent real lane length.
    3. We evaluate both assignments against expected real ratio ``lane_length_ft / lane_width_ft``.
    4. The assignment with smaller ratio error is selected and used to build homography.

    Args:
        lane_polygon: ``(N, 2)`` polygon points in image pixel coordinates.
        lane_length_ft: Real lane length represented by the lane patch.
        lane_width_ft: Real lane width represented by the lane patch.

    Returns:
        ``(H, H_inv, feet_per_pixel_x, feet_per_pixel_y)`` where x-axis is lane length axis
        in top-down plane and y-axis is lane width axis.

    Raises:
        ValueError: If input is malformed or resulting dimensions are invalid.
    """
    if lane_polygon.ndim != 2 or lane_polygon.shape[1] != 2:
        raise ValueError(f"lane_polygon must be (N, 2), got {lane_polygon.shape}")

    rect = cv2.minAreaRect(lane_polygon.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)
    src = _order_points_clockwise(box)

    side1_px = float(np.linalg.norm(src[1] - src[0]))
    side2_px = float(np.linalg.norm(src[2] - src[1]))
    if side1_px <= 0.0 or side2_px <= 0.0:
        raise ValueError("Invalid rectangle dimensions computed from lane polygon.")

    expected_ratio = lane_length_ft / lane_width_ft

    ratio_a = side1_px / side2_px
    ratio_b = side2_px / side1_px

    diff_a = abs(ratio_a - expected_ratio)
    diff_b = abs(ratio_b - expected_ratio)

    if diff_a <= diff_b:
        src_used = src
        length_px = side1_px
        width_px = side2_px
    else:
        # Rotate source ordering so edge (0->1) corresponds to side2, i.e. lane length axis.
        src_used = np.roll(src, shift=-1, axis=0).astype(np.float32)
        length_px = side2_px
        width_px = side1_px

    dst = np.array(
        [
            [0.0, 0.0],
            [length_px, 0.0],
            [length_px, width_px],
            [0.0, width_px],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(src_used, dst)
    inverse_homography = cv2.getPerspectiveTransform(dst, src_used)

    feet_per_pixel_x = lane_length_ft / length_px
    feet_per_pixel_y = lane_width_ft / width_px

    return homography, inverse_homography, feet_per_pixel_x, feet_per_pixel_y
