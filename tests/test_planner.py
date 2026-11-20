"""The planner's simple cases: nobody around, near the goal, at the goal."""

import numpy as np
from g1.planner import ARRIVED, MAX_SPEED, SLOW_DOWN_GAIN, Planner

NOBODY = np.zeros((0, 4))


def test_with_nobody_around_it_walks_straight_at_full_speed():
    planner = Planner((5.0, 0.0))
    velocity = planner.plan(np.array([-5.0, 0.0]), NOBODY)
    assert np.allclose(velocity, [MAX_SPEED, 0.0])


def test_it_slows_down_near_the_goal():
    planner = Planner((5.0, 0.0))
    distance = 0.5
    velocity = planner.plan(np.array([5.0 - distance, 0.0]), NOBODY)
    assert 0.0 < np.hypot(*velocity) <= SLOW_DOWN_GAIN * distance + 1e-9


def test_once_arrived_it_stays_arrived():
    planner = Planner((5.0, 0.0))
    planner.plan(np.array([5.0 - ARRIVED / 2, 0.0]), NOBODY)
    assert planner.arrived
    velocity = planner.plan(np.array([0.0, 0.0]), NOBODY)  # pushed away: still done
    assert planner.arrived
    assert not np.any(velocity)
