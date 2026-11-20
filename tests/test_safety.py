"""The safety layer with raw lidar points, and the freshness rule of the safety node."""

import numpy as np
from fakes import scan_circles_and_boxes
from g1.config import PERSON_RADIUS
from g1.navigator import Navigator
from g1.planner import ROBOT_RADIUS
from g1.safety import UNCERTAINTY, SpeedSeparationMonitor, allowed_speed, sensors_fresh

NOBODY = np.zeros((0, 4))
FULL = (1.0, 0.0, 0.0)  # 1 m/s straight ahead (east)
ROBOT = (0.0, 0.0, 0.0)


def test_a_lidar_point_right_ahead_stops_the_robot():
    safety = SpeedSeparationMonitor()
    points = np.array([[ROBOT_RADIUS + 0.02, 0.0]])
    assert safety.limit(FULL, ROBOT, NOBODY, points) == (0.0, 0.0, 0.0)
    assert safety.limited


def test_a_lidar_point_behind_or_far_away_changes_nothing():
    safety = SpeedSeparationMonitor()
    assert safety.limit(FULL, ROBOT, NOBODY, np.array([[-0.5, 0.0]])) == FULL
    assert safety.limit(FULL, ROBOT, NOBODY, np.array([[5.0, 0.0]])) == FULL
    assert not safety.limited


def test_a_lidar_point_counts_as_something_standing_still():
    safety = SpeedSeparationMonitor()
    forward, _, _ = safety.limit(FULL, ROBOT, NOBODY, np.array([[1.0, 0.0]]))
    assert abs(forward - allowed_speed(1.0 - ROBOT_RADIUS - UNCERTAINTY, 0.0)) < 1e-9


def test_a_person_and_lidar_points_together_use_the_stricter_limit():
    safety = SpeedSeparationMonitor()
    person = np.array([[1.5, 0.0, 0.0, 0.0]])
    only_person, _, _ = safety.limit(FULL, ROBOT, person)
    both, _, _ = safety.limit(FULL, ROBOT, person, np.array([[1.0, 0.0]]))
    assert both < only_person < 1.0
    assert abs(only_person - allowed_speed(1.5 - ROBOT_RADIUS - PERSON_RADIUS - UNCERTAINTY, 0.0)) < 1e-9


def test_sensors_must_all_have_reported_recently():
    assert sensors_fresh(1.0, 0.9, 0.95)
    assert not sensors_fresh(1.0, None, 0.95)  # never reported
    assert not sensors_fresh(1.0, 0.9, 0.5)  # too old


def test_the_navigator_stops_for_a_wall_but_not_for_a_table_on_its_map():
    box = (0.4, -1.0, 1.4, 1.0)  # 0.4 m ahead, too wide to be a person
    ranges = scan_circles_and_boxes(ROBOT[:2], ROBOT[2], [], PERSON_RADIUS, [box])

    unknown = Navigator(goal=(5.0, 0.0))
    forward, _, _ = unknown.update(ROBOT, ranges, dt=0.02)
    assert forward == 0.0
    assert unknown.safety.limited

    on_the_map = Navigator(goal=(5.0, 0.0), static_map=[box])
    forward, _, _ = on_the_map.update(ROBOT, ranges, dt=0.02)
    assert forward > 0.0
    assert len(on_the_map.points) == 0
