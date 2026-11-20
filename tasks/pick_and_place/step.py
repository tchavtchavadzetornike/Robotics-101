from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    """One step of the pick-and-place plan."""

    name: str
    start: Callable[[], bool | None]  # begins the step; False means it can't be done
    finished: Callable[[], bool]  # true once the step is complete
