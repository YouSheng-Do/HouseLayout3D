import sys
from pathlib import Path

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lift import (_polygon_parts, horizontal_mesh, opening_rectangle,
                  repair_polygon, wall_meshes)


def test_concave_room_lifts_to_floor_ceiling_and_walls():
    polygon, status = repair_polygon(np.array(
        [[0, 0], [2, 0], [2, 1], [1, 0.5], [0, 1]], dtype=float))
    assert status == "unchanged"
    floor = horizontal_mesh(polygon, 0.1)
    assert len(floor.triangles) == 3
    assert np.allclose(np.asarray(floor.vertices)[:, 2], 0.1)
    walls = wall_meshes(polygon, 0.1, 2.6)
    assert len(walls) == 5
    assert all(len(mesh.vertices) == 4 and len(mesh.triangles) == 2 for mesh in walls)


def test_door_height_is_explicit_rectangle():
    vertices, _ = opening_rectangle([[1, 2], [2, 2]], 0.25, 2.35,
                                    np.array([0.0, 1.0, 0.0]))
    assert set(np.round(vertices[:, 2], 8)) == {0.25, 2.35}
    assert np.isclose(np.ptp(vertices[:, 2]), 2.10)


def test_nested_multipolygon_repair_parts_are_not_dropped():
    nested = GeometryCollection([MultiPolygon([
        Polygon([(0, 0), (1, 0), (0, 1)]),
        Polygon([(2, 2), (4, 2), (2, 4)]),
    ])])
    parts = _polygon_parts(nested)
    assert len(parts) == 2
    assert max(part.area for part in parts) == 2.0
