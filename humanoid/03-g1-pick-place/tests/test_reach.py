"""Where the robot can grasp and put down (g1/reach.py), without a simulator.

  uv run --with pytest pytest tests/test_reach.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1 import reach  # noqa: E402


def test_the_mission_bottle_is_grasped_from_the_front():
    options, _ = reach.grasp_options((1.42, 0.0, 0.875))
    assert options[0].side == "-x"
    assert np.allclose(options[0].stand, (1.12, 0.12, 0.0))  # where mission.py stands


def test_middle_of_the_table_is_out_of_reach():
    options, reason = reach.grasp_options((1.6, 0.0, 0.875))
    assert options == []
    assert "can't reach" in reason and "30 cm from the -x edge" in reason


def test_back_of_the_table_from_the_back():
    options, _ = reach.grasp_options((1.8, 0.0, 0.875))
    assert options[0].side == "+x" and np.isclose(options[0].heading, np.pi)


def test_neighbor_in_the_hands_way():
    bottle = (1.42, 0.0)
    neighbor = (1.42, -0.09)  # right next to it, on the hand's side seen from the front
    options, _ = reach.grasp_options((*bottle, 0.875), [neighbor])
    assert all(o.side != "-x" for o in options)


def test_mark_is_placed_from_the_front_of_table_b():
    options, _ = reach.place_options("mark")
    assert options and options[0].side == "+y"


def test_next_to_keeps_the_gap():
    anchor = (1.42, 0.0)
    options, _ = reach.place_options(("next_to", anchor), [anchor])
    assert options
    for o in options:
        assert any(np.isclose(np.hypot(*(o.spot - anchor)), r) for r in reach.NEXT_TO)


def test_place_avoids_other_objects():
    others = [(1.37 + 0.05 * i, 0.3) for i in range(10)]
    options, _ = reach.place_options("table_a", others)
    assert options
    assert all(reach.free_spot(o.spot, others) for o in options)


def test_fingers_sweeping_into_a_neighbor_count_as_blocked():
    # phase 4, room 1: placing at this spot from table B's +y side, the fingers pushed this box 6 cm
    spot, heading, box = (0.756, -1.461), -np.pi / 2, (0.688, -1.583)
    assert reach.blocked(spot, heading, [box]) is not None
    assert reach.blocked(spot, heading, [(1.2, -1.461)]) is None  # far to the robot's left: free


def test_placing_next_to_an_object_on_the_left_is_free():
    # placing comes straight down and pulls straight back: a neighbor 13 cm to the left is fine
    spot, heading = (1.42, 0.0), 0.0
    assert reach.blocked(spot, heading, [(1.42, 0.13)], reach.PLACE_SWEEP) is None
    assert reach.blocked(spot, heading, [(1.53, 0.0)], reach.PLACE_SWEEP) is not None  # fingers reach it
    # on the right, the carried object arrives 6 cm right of the spot (phase 4, room 0: a box pushed 8 cm)
    assert reach.blocked(spot, heading, [(1.38, -0.13)], reach.PLACE_SWEEP) is not None


def test_the_carrying_wrist_needs_room_on_the_right():
    # phase 5, agent case 7: the box carried in to (0.85, -1.5) from +y, the wrist pushed this bottle 25 cm
    spot, heading, bottle = (0.85, -1.5), -np.pi / 2, (0.70, -1.45)
    assert reach.blocked(spot, heading, [bottle], reach.PLACE_SWEEP) is not None
    options, _ = reach.place_options("table_b", [bottle])
    assert options
    assert all(reach.blocked(o.spot, o.heading, [bottle], reach.PLACE_SWEEP) is None for o in options)
