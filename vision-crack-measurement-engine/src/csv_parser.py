"""CSV parsing utilities for crack and lane annotations."""

from __future__ import annotations

import ast
from typing import List, Tuple

import numpy as np
import pandas as pd


LaneParseResult = Tuple[
    str,
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    int,
    int,
]


REQUIRED_COLUMNS = [
    "Image Name",
    "Image Width",
    "Image Height",
    "Crack Type",
    "Polygon",
]


def _parse_polygon(polygon_str: str) -> np.ndarray:
    """Safely parse a polygon string into an ``(N, 2)`` float array."""
    data = ast.literal_eval(polygon_str)
    polygon = np.asarray(data, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError(f"Invalid polygon shape: {polygon.shape}. Expected (N, 2).")
    return polygon


def parse_annotation_csv(csv_path: str) -> LaneParseResult:
    """Parse annotation CSV and split lane and crack polygons.

    The first image in the CSV is selected and returned, and all polygons are parsed
    from rows matching that image.

    Args:
        csv_path: Path to the annotation CSV file.

    Returns:
        A tuple of:
            - image name
            - lane polygon as ``np.ndarray`` of shape ``(N, 2)``
            - longitudinal crack polygons
            - block crack polygons
            - pothole polygons
            - alligator crack polygons
            - image width
            - image height

    Raises:
        ValueError: If required columns are missing, no rows exist, or lane polygon is missing.
    """
    # ``sep=None`` lets pandas infer comma vs tab delimiters.
    df = pd.read_csv(csv_path, sep=None, engine="python")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("No annotation rows found in CSV.")

    image_name = str(df.iloc[0]["Image Name"])
    df = df[df["Image Name"] == image_name]

    if df.empty:
        raise ValueError(f"No annotation rows found for image '{image_name}'.")

    image_width = int(df.iloc[0]["Image Width"])
    image_height = int(df.iloc[0]["Image Height"])

    lane_polygon: np.ndarray | None = None
    longitudinal_polygons: List[np.ndarray] = []
    block_polygons: List[np.ndarray] = []
    pothole_polygons: List[np.ndarray] = []
    alligator_polygons: List[np.ndarray] = []

    for _, row in df.iterrows():
        crack_type = str(row["Crack Type"]).strip()
        polygon = _parse_polygon(str(row["Polygon"]))

        if crack_type == "Lane Segmentor":
            if lane_polygon is not None:
                raise ValueError("Multiple Lane Segmentor polygons found. Expected exactly one.")
            lane_polygon = polygon
        elif crack_type == "Longitudinal Cracking":
            longitudinal_polygons.append(polygon)
        elif crack_type == "Block Cracking":
            block_polygons.append(polygon)
        elif crack_type == "Pothole":
            pothole_polygons.append(polygon)
        elif crack_type == "Alligator Crack":
            alligator_polygons.append(polygon)

    if lane_polygon is None:
        raise ValueError("Lane Segmentor polygon not found in CSV rows.")

    return (
        image_name,
        lane_polygon,
        longitudinal_polygons,
        block_polygons,
        pothole_polygons,
        alligator_polygons,
        image_width,
        image_height,
    )
