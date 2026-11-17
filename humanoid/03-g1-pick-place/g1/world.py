"""The robot's memory of objects: what it has seen, where, and when.

The camera only sees what is in front of it. The memory keeps everything the
robot has seen, so it can act on an object it saw a minute ago behind it.
Each update does three things:

  1. Match (data association): pair each new sighting with a remembered object.
     The closest pairs first, at most GATE apart, and only if the colors agree.
     (The best overall pairing is the Hungarian algorithm; closest-first is
     enough while objects stand at least 13 cm apart. Exercise in PHASE-3.md.)
  2. Update or add: a matched object's position moves toward the sighting, its
     kind and color are votes, so one bad frame doesn't change them. An
     unmatched sighting is a new object with a new id.
  3. Forget, carefully: an object is only counted as missing if the camera
     *should* see it now: inside the image, and nothing nearer in the depth
     image covers it. Out of view or hidden: keep it, unchanged. Missing
     MISSES_TO_FORGET times in a row: it was moved away, forget it.

Ids never change and are never reused, so "object 3" means the same thing to
every node, and later to the language model.

Standing or lying comes from the latest sighting that could tell (objects.py).
A lying sighting has no kind; it may match the same-colored object further away
(FALLEN_GATE), so a fallen bottle keeps its id and stays a bottle.

The object in the hand (hold / release, told by the skills) is special: it
moves with the hand, so sightings near it update it instead of making a new
object, and it is never counted as missing from where it used to stand.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

GATE = 0.08  # m: a sighting further than this from an object is something else
SMOOTHING = 0.3  # how far a position moves toward each new sighting (0..1)
MISSES_TO_FORGET = 3
IMAGE_MARGIN = 20  # pixels: near the image edge, an object may be half outside
HIDDEN = 0.05  # m: something this much nearer in the depth image covers the object
ARM_AREA = 0.3  # m: around where the arm works, the robot's own hand hides objects
FALLEN_GATE = 0.15  # m: a lying sighting may be this far from where the object stood (a bottle's center moves ~9 cm when it falls)
HELD_RADIUS = 0.2  # m: a sighting this close to the held object, same color, is the held object
MAX_RANGE = 2.5  # m: further away, the detector may miss it, don't count as missing
AXIS = (-0.05, -0.025, 0.0, 0.025, 0.05)  # m above / below the center: points checked up the object
ENOUGH = 3  # of the AXIS points must be in view and uncovered


@dataclass
class Remembered:
    id: int
    position: np.ndarray
    first_seen: float
    last_seen: float
    kinds: Counter = field(default_factory=Counter)
    colors: Counter = field(default_factory=Counter)
    misses: int = 0
    sightings: int = 0
    standing: bool = True  # from the latest sighting that could tell

    @property
    def kind(self):
        return self.kinds.most_common(1)[0][0] if self.kinds else None

    @property
    def color(self):
        return self.colors.most_common(1)[0][0] if self.colors else None

    @property
    def label(self):
        return f"{self.color} {self.kind or 'object'}"

    def as_dict(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "color": self.color,
            "position": [round(float(v), 4) for v in self.position],
            "last_seen": round(self.last_seen, 2),
            "sightings": self.sightings,
            "standing": self.standing,
        }


class World:
    def __init__(self):
        self.objects = {}
        self.next_id = 1
        self.held = None
        self.arm_area = None  # (x, y) where the arm works at a table, or None

    def arm_working(self, spot):
        """The arm works at a table around spot (x, y), or None when it is done.

        The robot's own hand stands right next to objects there, at their depth:
        the detector misses them, and nothing nearer covers them. So near the arm,
        a miss proves nothing and is not counted.
        """
        self.arm_area = None if spot is None else np.asarray(spot[:2], float)

    def hold(self, object_id):
        self.held = object_id if object_id in self.objects else None

    def release(self, position=None):
        """The held object stands somewhere again: at `position` if given."""
        if self.held is not None and position is not None:
            self.objects[self.held].position = np.asarray(position, float)
            self.objects[self.held].misses = 0
        self.held = None

    def _held_sighting(self, seen):
        """Index of the sighting that is the held object, or None."""
        if self.held is None:
            return None
        obj = self.objects[self.held]
        near = [
            (np.hypot(*(s.position[:2] - obj.position[:2])), i)
            for i, s in enumerate(seen)
            if s.color == obj.color and np.hypot(*(s.position[:2] - obj.position[:2])) < HELD_RADIUS
        ]
        return min(near)[1] if near else None

    def match(self, seen):
        """[(sighting index, object id)], closest pairs first."""
        pairs = []
        for i, s in enumerate(seen):
            for obj in self.objects.values():
                if obj.color is not None and s.color != obj.color:
                    continue
                distance = np.hypot(*(s.position[:2] - obj.position[:2]))
                if distance <= (FALLEN_GATE if s.standing is False else GATE):
                    pairs.append((distance, i, obj.id))
        used_seen, used_objects, matches = set(), set(), []
        for _, i, object_id in sorted(pairs):
            if i in used_seen or object_id in used_objects:
                continue
            used_seen.add(i)
            used_objects.add(object_id)
            matches.append((i, object_id))
        return matches

    def update(self, time, seen, should_see):
        """seen: g1.objects.Seen list from one image. should_see(position) -> bool for that image."""
        held_index = self._held_sighting(seen)
        if held_index is not None:
            self.objects[self.held].position = seen[held_index].position.copy()
            self.objects[self.held].last_seen = time
            seen = seen[:held_index] + seen[held_index + 1 :]
        matches = [(i, o) for i, o in self.match(seen) if o != self.held]
        matched_seen = {i for i, _ in matches}
        matched_objects = {object_id for _, object_id in matches} | {self.held}

        for i, object_id in matches:
            obj, s = self.objects[object_id], seen[i]
            if s.standing is not None and s.standing != obj.standing:
                obj.position = s.position.copy()  # it fell or was stood up: a jump, not noise
                obj.standing = s.standing
            else:
                obj.position = obj.position + SMOOTHING * (s.position - obj.position)
            obj.last_seen = time
            obj.misses = 0
            obj.sightings += 1
            if s.kind is not None:
                obj.kinds[s.kind] += 1
            obj.colors[s.color] += 1

        for i, s in enumerate(seen):
            if i in matched_seen:
                continue
            obj = Remembered(self.next_id, s.position.copy(), time, time, sightings=1, standing=s.standing is not False)
            if s.kind is not None:
                obj.kinds[s.kind] += 1
            obj.colors[s.color] += 1
            self.objects[obj.id] = obj
            self.next_id += 1

        for object_id in list(self.objects):
            if object_id in matched_objects:
                continue
            near_arm = self.arm_area is not None and np.hypot(*(self.objects[object_id].position[:2] - self.arm_area)) < ARM_AREA
            if not near_arm and should_see(self.objects[object_id].position):
                self.objects[object_id].misses += 1
                if self.objects[object_id].misses >= MISSES_TO_FORGET:
                    del self.objects[object_id]

    def find(self, kind=None, color=None):
        return [o for o in self.objects.values() if (kind is None or o.kind == kind) and (color is None or o.color == color)]

    def as_list(self):
        return [o.as_dict() for o in sorted(self.objects.values(), key=lambda o: o.id)]


def point_visible(point, depth, fx, fy, cx, cy, cam_position, cam_rotation):
    """Is this one point in the image, in range, and not covered by something nearer?"""
    x, y, z = cam_rotation.T @ (np.asarray(point) - cam_position)
    if z <= 0.1 or z > MAX_RANGE:
        return False
    u, v = fx * x / z + cx, fy * y / z + cy
    height, width = depth.shape
    if not (IMAGE_MARGIN <= u < width - IMAGE_MARGIN and IMAGE_MARGIN <= v < height - IMAGE_MARGIN):
        return False
    return depth[int(v), int(u)] > z - HIDDEN


def should_see(position, depth, fx, fy, cx, cy, cam_position, cam_rotation):
    """Would the camera see an object here now? Enough points up its axis are visible.

    One point is not enough: a bottle in front may cover the middle of the spot
    and leave its top free, or the other way round. If the spot is covered, the
    robot can't know whether the object is still there, so it keeps it.
    """
    visible = sum(
        point_visible(np.asarray(position) + [0.0, 0.0, dz], depth, fx, fy, cx, cy, cam_position, cam_rotation)
        for dz in AXIS
    )
    return visible >= ENOUGH
