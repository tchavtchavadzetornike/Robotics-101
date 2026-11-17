"""Paths around tables (g1/paths.py), without a simulator.

  uv run --with pytest pytest tests/test_paths.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1 import paths  # noqa: E402

BOXES = paths.obstacles()


def no_segment_enters_a_table(path):
    return all(paths.free(a, b, BOXES) for a, b in zip(path, path[1:]))


def test_straight_when_free():
    path = paths.plan((0.0, 0.0), (0.9, -0.7))
    assert len(path) == 2


def test_around_table_a_to_its_far_side():
    path = paths.plan((0.0, 0.0), (2.3, 0.0))  # behind table A
    assert len(path) > 2
    assert no_segment_enters_a_table(path)
    assert paths.length(path) > 2.3


def test_around_table_b():
    path = paths.plan((0.95, -0.7), (1.0, -2.45))  # behind table B
    assert path is not None and no_segment_enters_a_table(path)


def test_crosses():
    box = (0.0, 0.0, 1.0, 1.0)
    assert paths.crosses((-1, 0.5), (2, 0.5), box)
    assert not paths.crosses((-1, 1.5), (2, 1.5), box)
    assert not paths.crosses((-1, 0.0), (2, 0.0), box)  # along the edge: touching is not entering


def test_every_side_of_every_table_reachable_from_start():
    for x0, y0, x1, y1 in BOXES:
        for goal in ((x0 - 0.05, (y0 + y1) / 2), (x1 + 0.05, (y0 + y1) / 2), ((x0 + x1) / 2, y0 - 0.05), ((x0 + x1) / 2, y1 + 0.05)):
            if any(paths.inside(goal, b) for b in BOXES):
                continue
            path = paths.plan((0.0, 0.0), goal)
            assert path is not None and no_segment_enters_a_table(path), goal
            assert np.allclose(path[-1], goal)
