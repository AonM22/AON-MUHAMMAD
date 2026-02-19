"""CLI entrypoint for crack measurement pipeline."""

from __future__ import annotations

import argparse
import os

import cv2

try:
    from .crack_area import compute_total_polygon_area_sqft
    from .crack_length import compute_total_crack_length_ft
    from .csv_parser import parse_annotation_csv
    from .lane_scaling import compute_lane_scaling
    from .visualization import draw_and_save_result
except ImportError:
    from crack_area import compute_total_polygon_area_sqft
    from crack_length import compute_total_crack_length_ft
    from csv_parser import parse_annotation_csv
    from lane_scaling import compute_lane_scaling
    from visualization import draw_and_save_result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Vision crack measurement engine")
    parser.add_argument("--csv", required=True, help="Path to annotations CSV")
    parser.add_argument("--output", required=True, help="Path to output image")
    return parser.parse_args()


def main() -> None:
    """Run the full measurement pipeline."""
    args = parse_args()

    (
        image_name,
        lane_polygon,
        longitudinal_polygons,
        block_polygons,
        pothole_polygons,
        alligator_polygons,
        _,
        _,
    ) = parse_annotation_csv(args.csv)
    image_path = os.path.join("data", image_name)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    homography, inverse_homography, feet_per_pixel_x, feet_per_pixel_y = compute_lane_scaling(lane_polygon)

    grid_lines = []

    total_crack_length_ft = compute_total_crack_length_ft(
        longitudinal_polygons=longitudinal_polygons,
        homography=homography,
        feet_per_pixel_x=feet_per_pixel_x,
        feet_per_pixel_y=feet_per_pixel_y,
    )

    block_area_sqft = compute_total_polygon_area_sqft(
        polygons=block_polygons,
        homography=homography,
        feet_per_pixel_x=feet_per_pixel_x,
        feet_per_pixel_y=feet_per_pixel_y,
    )
    pothole_area_sqft = compute_total_polygon_area_sqft(
        polygons=pothole_polygons,
        homography=homography,
        feet_per_pixel_x=feet_per_pixel_x,
        feet_per_pixel_y=feet_per_pixel_y,
    )
    alligator_area_sqft = compute_total_polygon_area_sqft(
        polygons=alligator_polygons,
        homography=homography,
        feet_per_pixel_x=feet_per_pixel_x,
        feet_per_pixel_y=feet_per_pixel_y,
    )

    total_cracks_area_sqft = block_area_sqft + pothole_area_sqft + alligator_area_sqft

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    draw_and_save_result(
        image=image,
        lane_polygon=lane_polygon,
        longitudinal_polygons=longitudinal_polygons,
        block_polygons=block_polygons,
        pothole_polygons=pothole_polygons,
        alligator_polygons=alligator_polygons,
        grid_lines=grid_lines,
        longitudinal_length_ft=total_crack_length_ft,
        block_area_sqft=block_area_sqft,
        pothole_area_sqft=pothole_area_sqft,
        alligator_area_sqft=alligator_area_sqft,
        total_cracks_area_sqft=total_cracks_area_sqft,
        output_path=args.output,
    )

    print(f"Block Cracking (sq ft): {block_area_sqft:.2f}")
    print(f"Longitudinal Cracking (ft): {total_crack_length_ft:.2f}")
    print(f"Pothole (sq ft): {pothole_area_sqft:.2f}")
    print(f"Alligator Crack (sq ft): {alligator_area_sqft:.2f}")
    print(f"Total cracks area (sq ft): {total_cracks_area_sqft:.2f}")


if __name__ == "__main__":
    main()
