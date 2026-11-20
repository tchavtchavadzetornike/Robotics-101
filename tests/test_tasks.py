"""What every task folder provides, and the crowd's size limit."""

from types import SimpleNamespace

import pytest
from area.crowd import Crowd

import tasks


def cfg():
    return SimpleNamespace(seed=None, events=SimpleNamespace(move_crowd=SimpleNamespace(params={"seed": 0})))


def test_configure_sets_the_crowd_seed_for_crossing():
    config = cfg()
    tasks.configure(tasks.crossing.ENV_ID, config, 7)
    assert config.seed == 7
    assert config.events.move_crowd.params["seed"] == 7


def test_configure_sets_the_seed_for_pick_and_place():
    config = cfg()
    tasks.configure(tasks.pick_and_place.ENV_ID, config, 3)
    assert config.seed == 3


def test_configure_rejects_an_unknown_environment():
    with pytest.raises(KeyError):
        tasks.configure("G1-Nothing-v0", cfg(), 0)


def test_a_crowd_too_big_for_its_square_says_so_instead_of_hanging():
    with pytest.raises(ValueError, match="too many people"):
        Crowd(count=60, half_size=2.0, seed=0)
