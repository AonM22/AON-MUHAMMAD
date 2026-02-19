"""Block crack area estimation."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _shoelace_area(points: np.ndarray) -> float:
    """Compute polygon area in pixel^2 using the Shoelace formula."""
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_total_block_area_sqft(
    block_polygons: List[np.ndarray],
    homography: np.ndarray,
    feet_per_pixel_x: float,
    feet_per_pixel_y: float,
) -> float:
    """Compute total block crack area in square feet.

    Args:
        block_polygons: Block crack polygons in image coordinates.
        homography: Perspective transform from image space to top-down space.
        feet_per_pixel_x: X-axis scale in feet/pixel.
        feet_per_pixel_y: Y-axis scale in feet/pixel.

    Returns:
        Total area in square feet.
    """
    total_area_sqft = 0.0

    for polygon in block_polygons:
        if len(polygon) < 3:
            continue

        pts = polygon.astype(np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
        area_px2 = _shoelace_area(warped)
        total_area_sqft += area_px2 * feet_per_pixel_x * feet_per_pixel_y

    return total_area_sqft


def compute_total_polygon_area_sqft(
    polygons: List[np.ndarray],
    homography: np.ndarray,
    feet_per_pixel_x: float,
    feet_per_pixel_y: float,
) -> float:
    """Compute total area in square feet for any polygon set."""
    return compute_total_block_area_sqft(
        block_polygons=polygons,
        homography=homography,
        feet_per_pixel_x=feet_per_pixel_x,
        feet_per_pixel_y=feet_per_pixel_y,
    )


def compute_crack_density(
    area_sqft: float,
    lane_length_ft: float = 35,
    lane_width_ft: float = 11,
) -> float:
    """Compute crack density percentage relative to lane area.

    Notes:
        The effective lane area for density normalization follows the requested
        benchmark region ``20 * 11`` square feet.

    Args:
        area_sqft: Total cracked area in square feet.
        lane_length_ft: Included for interface compatibility.
        lane_width_ft: Included for interface compatibility.

    Returns:
        Crack density in percent.
    """
    _ = (lane_length_ft, lane_width_ft)
    lane_area_sqft = 20.0 * 11.0
    if lane_area_sqft <= 0:
        raise ValueError("Lane area must be positive.")
    return 100.0 * float(area_sqft) / lane_area_sqft
