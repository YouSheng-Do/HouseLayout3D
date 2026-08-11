import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import density_image, opening_segment, pixels_to_world, world_to_annotation_pixels


def test_density_and_annotation_roundtrip():
    rng = np.random.default_rng(7)
    points = rng.uniform([-3.0, 2.0, 0.0], [8.0, 7.0, 3.0], size=(10000, 3))
    density, transform = density_image(points)
    assert density.shape == (256, 256)
    assert density.dtype == np.float32
    assert np.isclose(density.max(), 1.0)
    pixels = world_to_annotation_pixels(points[:, :2], transform)
    recovered = pixels_to_world(pixels, transform)
    max_error = np.max(np.abs(recovered - points[:, :2]))
    assert max_error <= transform["max_range"] / 256 + 1e-8


def test_opening_rectangle_collapses_to_long_xy_edge():
    vertices = [[1, 2, 0], [4, 2, 0], [4, 2, 2], [1, 2, 2]]
    segment = opening_segment(vertices)
    assert {tuple(x) for x in segment} == {(1.0, 2.0), (4.0, 2.0)}


def test_opening_match_counts_fp_on_gt_empty_aligned_level():
    from evaluate import match_openings_by_level
    pred = [{"level": 0, "seg": np.array([[0, 0], [1, 0]])}]
    assert match_openings_by_level(pred, [], {0: 0.0}, {0: 0.0}, 0.5) == [0, 1, 0]
