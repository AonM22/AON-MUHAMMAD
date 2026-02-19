"""Visualization overlays for lane, cracks, grid, and metrics."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


LineSegment = Tuple[Tuple[int, int], Tuple[int, int]]


def _draw_polygon(image: np.ndarray, polygon: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    """Draw a single polygon outline on an image."""
    if len(polygon) < 2:
        return
    pts = polygon.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


def draw_and_save_result(
    image: np.ndarray,
    lane_polygon: np.ndarray,
    longitudinal_polygons: List[np.ndarray],
    block_polygons: List[np.ndarray],
    pothole_polygons: List[np.ndarray],
    alligator_polygons: List[np.ndarray],
    grid_lines: List[LineSegment],
    longitudinal_length_ft: float,
    block_area_sqft: float,
    pothole_area_sqft: float,
    alligator_area_sqft: float,
    total_cracks_area_sqft: float,
    output_path: str,
) -> np.ndarray:
    """Overlay annotations and metrics on image and save to disk.

    Colors (BGR):
        Lane polygon: green
        Longitudinal cracks: red
        Block cracks: blue
        Grid: light gray
    """
    canvas = image.copy()

    _draw_polygon(canvas, lane_polygon, color=(0, 255, 0), thickness=3)

    for polygon in longitudinal_polygons:
        _draw_polygon(canvas, polygon, color=(0, 0, 255), thickness=2)

    for polygon in block_polygons:
        _draw_polygon(canvas, polygon, color=(255, 0, 0), thickness=2)

    for polygon in pothole_polygons:
        _draw_polygon(canvas, polygon, color=(0, 255, 255), thickness=2)

    for polygon in alligator_polygons:
        _draw_polygon(canvas, polygon, color=(255, 0, 255), thickness=2)

    text_lines = [
        f"Block Cracking: {block_area_sqft:.2f} sq ft",
        f"Longitudinal Cracking: {longitudinal_length_ft:.2f} ft",
        f"Pothole: {pothole_area_sqft:.2f} sq ft",
        f"Alligator Crack: {alligator_area_sqft:.2f} sq ft",
        f"Total cracks area: {total_cracks_area_sqft:.2f} sq ft",
    ]

    x, y0 = 24, 40
    line_h = 30
    box_w = 560
    box_h = 170

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - 12, y0 - 28), (x - 12 + box_w, y0 - 28 + box_h), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)

    for i, line in enumerate(text_lines):
        y = y0 + i * line_h
        cv2.putText(
            canvas,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, canvas)
    return canvas
