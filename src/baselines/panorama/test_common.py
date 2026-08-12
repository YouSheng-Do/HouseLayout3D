import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (aligned_points_to_world, clean_polygon,
                    dopnet_xyz_to_local_floor, horizon_uv_to_local_floor)


def test_identity_floor_projection():
    pano = {"position": np.array([10.0, 20.0, 2.0]), "camera_height": 1.5}
    pose = np.eye(4)
    # pose_1_5 columns (right, down, forward) = (world +x, -z, +y),
    # and its forward is the panorama's left direction.
    pose[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    # Panorama right/forward/up maps to -pose-forward/pose-right/-pose-down.
    local = np.array([[1.0, 2.0, -1.5], [-2.0, 3.0, -1.5]])
    world = aligned_points_to_world(local, pano, pose, np.eye(3))
    expected = np.array([[12.0, 19.0, 0.5], [13.0, 22.0, 0.5]])
    np.testing.assert_allclose(world, expected)


def test_horizon_floor_rays_have_metric_height():
    # Three ceiling/floor pairs; floor is 45 degrees below horizon.
    uv = [[0.0, 0.25], [0.0, 0.75],
          [0.25, 0.25], [0.25, 0.75],
          [0.5, 0.25], [0.5, 0.75]]
    points = horizon_uv_to_local_floor(uv, 1.6)
    np.testing.assert_allclose(points[:, 2], -1.6)


def test_dopnet_axis_and_scale():
    xyz = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
    local = dopnet_xyz_to_local_floor(xyz, 1.6)
    np.testing.assert_allclose(local[0], [1.6, 3.2, -1.6])


def test_polygon_repair_and_orientation():
    ring = clean_polygon(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
    assert ring is not None
    area2 = np.sum(ring[:, 0] * np.roll(ring[:, 1], -1)
                   - np.roll(ring[:, 0], -1) * ring[:, 1])
    assert area2 > 0
