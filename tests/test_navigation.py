"""Tests for the robot's navigation software (package g1).

Plain Python + NumPy: they run on any computer, without Isaac Sim or dora.

    pip install -e source/g1 pytest
    pytest tests
"""

import numpy as np
from g1 import messages
from g1.config import SCAN_DT
from g1.planner import ROBOT_RADIUS, Planner
from g1.tracker import Tracker

PERSON_RADIUS = 0.22  # m


def scan_of(robot, people, heading=0.0, rays=360):
    """A perfect lidar scan from `robot` (x, y) of round people at `people`."""
    angles = np.radians(np.arange(rays) * 360.0 / rays) + heading
    ranges = np.full(rays, np.inf)
    for px, py in people:
        dx, dy = px - robot[0], py - robot[1]
        along = dx * np.cos(angles) + dy * np.sin(angles)  # distance along each ray
        miss2 = dx * dx + dy * dy - along**2  # squared distance of the ray to the center
        hit = (along > 0) & (miss2 < PERSON_RADIUS**2)
        ranges = np.where(hit, np.minimum(ranges, along - np.sqrt(np.maximum(PERSON_RADIUS**2 - miss2, 0))), ranges)
    return ranges


def test_tracker_follows_a_walking_person():
    tracker = Tracker()
    robot = (0.0, 0.0, 0.0)
    for k in range(20):  # 2 s: a person 3 m ahead walks to the left at 1 m/s
        person = (3.0, -1.0 + 1.0 * k * SCAN_DT)
        tracks = tracker.update(scan_of(robot[:2], [person]), robot)
    assert len(tracks) == 1
    x, y, vx, vy = tracks[0]
    assert np.hypot(x - person[0], y - person[1]) < 0.1  # where the person is, within 10 cm
    assert abs(vx) < 0.2 and abs(vy - 1.0) < 0.2  # and how fast they walk


def test_planner_walks_to_the_goal_around_a_person():
    planner = Planner((5.0, 0.0))
    position, heading = np.array([-5.0, 0.0]), 0.0
    person = np.array([0.0, 0.05])  # standing right on the straight line
    dt, closest = 0.02, np.inf
    for step in range(int(40 / dt)):
        if step % 5 == 0:  # plans 10 times per second
            planner.plan(position, np.array([[*person, 0.0, 0.0]]))
        forward, left, turn = planner.to_cmd_vel(heading, dt)
        c, s = np.cos(heading), np.sin(heading)
        position = position + dt * np.array([c * forward - s * left, s * forward + c * left])
        heading += dt * turn
        closest = min(closest, np.linalg.norm(position - person) - PERSON_RADIUS - ROBOT_RADIUS)
        if planner.arrived:
            break
    assert planner.arrived
    assert closest > 0.0  # never touched the person


def test_messages_round_trip():
    time, xy, heading = messages.unpack_pose(messages.pack_pose(1.5, 2.0, -3.0, 0.25))
    assert time == 1.5 and list(xy) == [2.0, -3.0] and heading == 0.25
    tracks = np.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4]])
    _, back = messages.unpack_tracks(messages.pack_tracks(0.1, tracks))
    assert np.array_equal(back, tracks)
    _, ranges = messages.unpack_scan(messages.pack_scan(0.2, np.array([1.0, np.inf])))
    assert ranges[0] == 1.0 and np.isinf(ranges[1])


def test_safety_stops_slows_and_lets_go():
    from g1.planner import MAX_DECEL
    from g1.safety import REACTION_TIME, UNCERTAINTY, SpeedSeparationMonitor, allowed_speed

    safety = SpeedSeparationMonitor()
    full = (1.0, 0.0, 0.0)  # 1 m/s straight ahead (east)
    robot = (0.0, 0.0, 0.0)
    # nobody around: full speed
    assert safety.limit(full, robot, np.zeros((0, 4))) == full
    # someone standing right in front, touching distance: stop (turning stays allowed)
    near = np.array([[ROBOT_RADIUS + PERSON_RADIUS + 0.02, 0.0, 0.0, 0.0]])
    assert safety.limit((1.0, 0.0, 0.5), robot, near) == (0.0, 0.0, 0.5)
    # someone right behind: walking away from them is always fine
    behind = np.array([[-(ROBOT_RADIUS + PERSON_RADIUS + 0.02), 0.0, 0.0, 0.0]])
    assert safety.limit(full, robot, behind) == full
    # someone 2 m ahead walking toward the robot at 1.3 m/s: slowed down to exactly
    # the speed the ISO/TS 15066 rule allows
    ahead = np.array([[2.0, 0.0, -1.3, 0.0]])
    forward, _, _ = safety.limit(full, robot, ahead)
    gap = 2.0 - ROBOT_RADIUS - PERSON_RADIUS - UNCERTAINTY
    assert 0.0 < forward < 1.0
    assert abs(forward - allowed_speed(gap, 1.3)) < 1e-9
    # and at that speed the rule holds: gap = v_h (T_r + T_s) + v T_r + v^2 / (2a)
    t_stop = forward / MAX_DECEL
    needed = 1.3 * (REACTION_TIME + t_stop) + forward * REACTION_TIME + forward**2 / (2 * MAX_DECEL)
    assert abs(needed - gap) < 1e-9
