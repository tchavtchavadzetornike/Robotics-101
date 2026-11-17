"""What is in a detector's box: where it stands, what kind it is, what color.

The detector (OWLv2, see tell-the-robot/PHASE-3.md) is good at *where*: it
finds 97% of the objects with tight boxes. It is bad at *what*: it calls every
bottle a cylinder. So the box only says where to look, and the rest comes from
the depth image and the colors inside the box:

  points  back-project every pixel in the box, keep what stands above the table
          and belongs to the nearest cluster (the box also covers table and background)
  kind    the object's top above the table: bottle 18 cm (with the cap), can 12 cm, box 10 cm
  color   the hue of its saturated pixels, the nearest of the four colors
  center  round kinds: fit a circle to the side (perception.cylinder_center);
          box: the middle of its top face. Height: the table plus half the object
  posture standing, or lying on its side: every standing object is at least 10 cm
          high, every lying one at most 6.8 cm (measured, spikes/fallen_check.py).
          A lying object's height says nothing about its kind (all about 6 cm), so its
          kind is unknown here; the memory keeps the kind it saw while it stood.

Unknown kind and posture: when the box touches the image's top edge, the top may be cut off.
"""

from dataclasses import dataclass

import numpy as np

from g1 import perception, scenes

TABLE_TOP = scenes.TABLE_TOP
ABOVE_TABLE = 0.008  # m: points lower than this over the table are the table itself
MAX_HEIGHT = 0.30  # m above the table: higher points are not on the table
CLUSTER = 0.06  # m: points this far from the cluster's middle belong to something else
MIN_POINTS = 40
EDGE = 2  # pixels
TOPS = {"bottle": 0.18, "can": 0.12, "box": 0.10}  # m, highest point above the table
LYING_BELOW = 0.08  # m: a top lower than this is an object on its side (standing >= 10 cm, lying <= 6.8 cm)
LYING_CENTER = 0.03  # m above the table: the axis of a cylinder lying on its side
HUES = {"red": 0.0, "yellow": 50.0, "green": 130.0, "blue": 210.0}  # degrees
MIN_SATURATION = 0.45


@dataclass
class Seen:
    kind: str | None  # None: top cut off, can't tell
    color: str
    position: np.ndarray  # center, room frame
    points: int
    box: tuple  # (u1, v1, u2, v2) in the image
    standing: bool | None = True  # False: lying on its side; None: top cut off, can't tell
    top: float = 0.0  # m, highest point above the table

    @property
    def label(self):
        return f"{self.color} {self.kind or 'object'}" + ("" if self.standing is not False else " (lying)")


def nearest_kind(top):
    return min(TOPS, key=lambda kind: abs(TOPS[kind] - top))


def nearest_color(hues):
    """Circular mean of the hues (0 and 360 are both red), then the nearest named color."""
    angles = np.radians(hues)
    mean = np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())) % 360

    def distance(name):
        d = abs(mean - HUES[name]) % 360
        return min(d, 360 - d)

    return min(HUES, key=distance)


def main_cluster(points):
    """The points near the middle of the biggest group, in the floor plan (x, y)."""
    middle = np.median(points[:, :2], axis=0)
    for _ in range(3):
        near = np.hypot(*(points[:, :2] - middle).T) < CLUSTER
        if near.sum() == 0:
            return near
        middle = np.median(points[near, :2], axis=0)
    return near


def describe(box, rgb, depth, fx, fy, cx, cy, cam_position, cam_rotation):
    """A detector box -> Seen, or None if too little of an object is in it."""
    height, width = depth.shape
    u1, v1 = max(int(box[0]), 0), max(int(box[1]), 0)
    u2, v2 = min(int(np.ceil(box[2])), width), min(int(np.ceil(box[3])), height)
    v, u = np.mgrid[v1:v2, u1:u2]
    v, u = v.ravel(), u.ravel()
    world = perception.to_world(
        perception.back_project(u, v, depth[v, u], fx, fy, cx, cy), cam_position, cam_rotation
    )
    above = world[:, 2] - TABLE_TOP
    keep = (above > ABOVE_TABLE) & (above < MAX_HEIGHT)
    world, u, v = world[keep], u[keep], v[keep]
    if len(world) < MIN_POINTS:
        return None
    near = main_cluster(world)
    world, u, v = world[near], u[near], v[near]
    if len(world) < MIN_POINTS:
        return None

    top_cut_off = v.min() < EDGE
    top = np.percentile(world[:, 2], 99) - TABLE_TOP
    standing = None if top_cut_off else bool(top >= LYING_BELOW)
    kind = nearest_kind(top) if standing else None

    hue, saturation, _ = perception.hsv(rgb[v, u][None])
    colored = saturation[0] > MIN_SATURATION
    color = nearest_color(hue[0][colored]) if colored.any() else "unknown"

    if standing is False:
        xy = world[:, :2].mean(axis=0)
        position = np.array([xy[0], xy[1], TABLE_TOP + LYING_CENTER])
        return Seen(None, color, position, len(world), (u1, v1, u2, v2), False, float(top))
    if kind == "box":
        top_face = world[world[:, 2] > world[:, 2].max() - 0.01]
        xy = top_face[:, :2].mean(axis=0)
    else:
        radius = scenes.SHAPES[kind or "bottle"][0]
        side = world[world[:, 2] < world[:, 2].max() - 0.02]
        xy = perception.cylinder_center(side if len(side) > 10 else world, cam_position, radius)
    half = scenes.SHAPES[kind or "bottle"][1]
    position = np.array([xy[0], xy[1], TABLE_TOP + half])
    return Seen(kind, color, position, len(world), (u1, v1, u2, v2), standing, float(top))


SAME_OBJECT = 0.05  # m: two sightings in one image closer than this are one object


def on_a_table(position):
    """Objects stand on tables: anything else (the robot's own hand, a table leg) is not one."""
    for (cx, cy), (hx, hy) in scenes.TABLES.values():
        if abs(position[0] - cx) <= hx and abs(position[1] - cy) <= hy:
            return True
    return False


def merge_same(seen):
    """One object per place: of sightings closer than SAME_OBJECT, keep the one with most points.

    The detector sometimes gives two boxes for one object (the whole bottle and
    just its top) that overlap too little to be merged as boxes. In 3D they
    land on the same spot.
    """
    kept = []
    for s in sorted(seen, key=lambda s: -s.points):
        if all(np.hypot(*(s.position[:2] - k.position[:2])) >= SAME_OBJECT for k in kept):
            kept.append(s)
    return kept


def look(detector, rgb, depth, fx, fy, cx, cy, cam_position, cam_rotation):
    """Everything on the tables in one image: detector boxes -> described, filtered, merged."""
    boxes, _ = detector(rgb)
    seen = [describe(b, rgb, depth, fx, fy, cx, cy, cam_position, cam_rotation) for b in boxes]
    real = [s for s in seen if s is not None and on_a_table(s.position) and s.color in HUES]
    return merge_same(real)  # no color: the robot's own (white and grey) hand, not an object


def to_dict(seen):
    return {
        "kind": seen.kind,
        "color": seen.color,
        "position": [round(float(v), 4) for v in seen.position],
        "points": int(seen.points),
        "box": [int(v) for v in seen.box],
        "standing": seen.standing,
        "top": round(float(seen.top), 3),
    }


def from_dict(d):
    return Seen(d["kind"], d["color"], np.array(d["position"]), d["points"], tuple(d["box"]), d.get("standing", True), d.get("top", 0.0))
