"""The tracker: lidar scans -> the people, and how each one moves."""

import numpy as np
from fakes import scan_circles_and_boxes
from g1.config import PERSON_RADIUS, SCAN_DT
from g1.tracker import CONFIRM_HITS, MAX_MISSES, Tracker, detect_people

ROBOT = (0.0, 0.0, 0.0)


def scan(*people):
    return scan_circles_and_boxes(ROBOT[:2], ROBOT[2], people, PERSON_RADIUS, [])


def test_a_person_across_the_first_ray_is_one_person():
    # straight ahead, the person's rays are on both sides of ray 0: the scan is a circle
    people = detect_people(scan((3.0, 0.0)), ROBOT)
    assert len(people) == 1
    assert np.hypot(people[0][0] - 3.0, people[0][1]) < 0.1


def test_something_wider_than_a_person_is_not_one():
    wall = [(2.0, -1.0, 2.2, 1.0)]
    ranges = scan_circles_and_boxes(ROBOT[:2], ROBOT[2], [], PERSON_RADIUS, wall)
    assert len(detect_people(ranges, ROBOT)) == 0


def test_a_person_counts_only_after_being_seen_a_few_times():
    tracker = Tracker()
    for _ in range(CONFIRM_HITS - 1):
        assert len(tracker.update(scan((3.0, 0.0)), ROBOT)) == 0
    assert len(tracker.update(scan((3.0, 0.0)), ROBOT)) == 1


def test_two_people_keep_their_own_velocities():
    tracker = Tracker()
    for k in range(20):  # 2 s: one walks to the left, the other to the right
        tracks = tracker.update(scan((3.0, -1.0 + k * SCAN_DT), (4.0, 2.0 - k * SCAN_DT)), ROBOT)
    assert len(tracks) == 2
    first, second = sorted(tracks, key=lambda track: track[0])
    assert abs(first[3] - 1.0) < 0.25
    assert abs(second[3] + 1.0) < 0.25


def test_a_person_out_of_sight_is_remembered_for_a_while_then_forgotten():
    tracker = Tracker()
    for _ in range(5):
        tracker.update(scan((3.0, 0.0)), ROBOT)
    for _ in range(MAX_MISSES):
        assert len(tracker.update(scan(), ROBOT)) == 1
    assert len(tracker.update(scan(), ROBOT)) == 0
