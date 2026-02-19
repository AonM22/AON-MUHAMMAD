# Vision Crack Measurement Engine

## Overview

`vision-crack-measurement-engine` converts segmentation polygon outputs into real-world pavement metrics.
Segmentation outputs are pixel-domain shapes, which are not directly usable for engineering measurement.
This project estimates lane scale from the lane polygon, rectifies geometry using homography, and reports
class-wise crack measurements in feet and square feet.

This repository demonstrates real-world geometric measurement from segmentation outputs using synthetic/sample data.

## Input Source and Data Contract

The `annotations.csv` file is expected to be the output of lane segmentation and road/crack segmentation models.
Each row represents one detected polygon.

Required CSV columns:

- `Image Name`
- `Image Width`
- `Image Height`
- `Crack Type`
- `Polygon`

Supported `Crack Type` values:

- `Lane Segmentor` (exactly one per image)
- `Longitudinal Cracking`
- `Block Cracking`
- `Pothole`
- `Alligator Crack`

`Image Name` is used to resolve the input image automatically from `data/<Image Name>`.

## Final Measurement Setup

- Lane reference dimensions used for scaling:
  - Length/height: `15 ft`
  - Width: `12 ft`
- Visualization:
  - Lane and crack polygons are shown
  - Grid lines are intentionally hidden in the final output
- Reported metrics:
  - Block Cracking area (`sq ft`)
  - Longitudinal Cracking length (`ft`)
  - Pothole area (`sq ft`)
  - Alligator Crack area (`sq ft`)
  - Total cracks area (`sq ft`)

## Architecture Flow

```text
Image + annotations.csv (model outputs)
        |
        v
CSV Parsing + Class Split
        |
        v
Lane Polygon Scaling (15 ft x 12 ft)
        |
        v
Homography to Top-Down Lane Plane
        |
        v
Warp Crack Polygons to Metric Space
        |
        v
Length/Area Computation by Class
        |
        v
Aggregate Total Cracks Area
        |
        v
Visualization + Console Metrics
```

See `architecture.png` for the final visual architecture.

## Mathematical Notes

1. Min-area rectangle:
   - `cv2.minAreaRect` is applied on lane polygon points.
   - Side assignment is selected by ratio matching against expected lane aspect ratio.
2. Homography:
   - `cv2.getPerspectiveTransform` maps perspective lane geometry to a top-view metric plane.
3. Area:
   - Polygon area is computed using the Shoelace formula after warping.
4. Length:
   - Longitudinal cracking length is computed in transformed space using metric scaling.
   - Region-like longitudinal polygons use dominant axis estimation for stable length output.

## Repository Structure

```text
vision-crack-measurement-engine/
|-- data/
|   |-- sample_image.jpg
|   `-- annotations.csv
|-- src/
|   |-- csv_parser.py
|   |-- lane_scaling.py
|   |-- crack_length.py
|   |-- crack_area.py
|   |-- visualization.py
|   `-- main.py
|-- tests/
|-- output/
|-- architecture.png
|-- requirements.txt
`-- README.md
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.main --csv data/annotations.csv --output output/result.png
```

## Example Output Metrics

```text
Block Cracking (sq ft): XX.XX
Longitudinal Cracking (ft): XX.XX
Pothole (sq ft): XX.XX
Alligator Crack (sq ft): XX.XX
Total cracks area (sq ft): XX.XX
```

## Engineering Notes

- CSV delimiter can be comma or tab; parser auto-detects both.
- The first image in CSV is selected as the active image context.
- The module design is intentionally split for parser, geometry, metrics, and rendering to keep production maintenance straightforward.
