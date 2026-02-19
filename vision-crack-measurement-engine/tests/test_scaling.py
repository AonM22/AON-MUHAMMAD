import numpy as np

from src.lane_scaling import compute_lane_scaling


def test_compute_lane_scaling_shapes():
    lane_polygon = np.array([[100, 100], [300, 100], [320, 180], [120, 180]], dtype=np.float32)
    h, h_inv, fpx, fpy = compute_lane_scaling(lane_polygon)

    assert h.shape == (3, 3)
    assert h_inv.shape == (3, 3)
    assert fpx > 0
    assert fpy > 0
