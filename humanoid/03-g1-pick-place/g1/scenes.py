"""Random rooms: the same tables, the target bottle, and 2 to 5 other objects.

A scene is made from robot/scene.xml: its bottle stays (the mission's target,
somewhere on table A where the right hand reaches it) and other objects are
added on both tables. Every object is 5 to 7 cm wide and stands upright, so
the existing side grasp works on all of them.

    scene = make_scene(seed=7)          # same seed, same room
    path = write_scene(scene, robot_dir) # robot/scene_seed7.xml, for MuJoCo
    scene.objects                        # the truth: name, kind, color, position

The file has to sit next to g1_with_hands.xml: MuJoCo finds included files
and meshes relative to the main file. So it goes into robot/ as
scene_seed<N>.xml, which git ignores.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COLORS = {
    "blue": (0.2, 0.55, 0.9),
    "red": (0.85, 0.15, 0.15),
    "green": (0.2, 0.7, 0.3),
    "yellow": (0.95, 0.8, 0.1),
}
# kind: (radius or half width, half height) in meters
SHAPES = {
    "bottle": (0.03, 0.075),
    "can": (0.033, 0.06),
    "box": (0.03, 0.05),
}
TABLES = {  # center (x, y) and half size of the top, from robot/scene.xml
    "table_a": ((1.6, 0.0), (0.3, 0.4)),
    "table_b": ((1.0, -1.7), (0.4, 0.3)),
}
TABLE_TOP = 0.80
EDGE_MARGIN = 0.07  # m: objects stay this far from a table edge
MIN_GAP = 0.13  # m between object centers
TARGET_X = (1.38, 1.46)  # where the target bottle may stand on table A: the hand reaches it
TARGET_Y = (-0.12, 0.12)
PLACE_MARK = (1.0, -1.52)
MARK_CLEARANCE = 0.15  # m: nothing stands on or near the green mark


@dataclass
class SceneObject:
    name: str
    kind: str
    color: str
    position: np.ndarray  # center, room frame

    @property
    def label(self):
        return f"{self.color} {self.kind}"


@dataclass
class Scene:
    seed: int
    objects: list = field(default_factory=list)

    def object(self, name):
        return next(o for o in self.objects if o.name == name)


def _free(position, objects):
    return all(np.hypot(*(position[:2] - o.position[:2])) >= MIN_GAP for o in objects)


def _spot_on(table, rng):
    (cx, cy), (hx, hy) = TABLES[table]
    return np.array(
        [
            rng.uniform(cx - hx + EDGE_MARGIN, cx + hx - EDGE_MARGIN),
            rng.uniform(cy - hy + EDGE_MARGIN, cy + hy - EDGE_MARGIN),
        ]
    )


def make_scene(seed, extra=None):
    """The target bottle on table A plus `extra` other objects (2-5 if None)."""
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, 6)) if extra is None else extra
    target_xy = np.array([rng.uniform(*TARGET_X), rng.uniform(*TARGET_Y)])
    objects = [
        SceneObject("bottle", "bottle", "blue", np.array([*target_xy, TABLE_TOP + SHAPES["bottle"][1]]))
    ]
    kinds = list(SHAPES)
    colors = list(COLORS)
    while len(objects) < count + 1:
        kind = kinds[rng.integers(len(kinds))]
        color = colors[rng.integers(len(colors))]
        if kind == "bottle" and color == "blue":
            continue  # only one blue bottle: "the blue bottle" must mean one thing
        table = "table_a" if rng.random() < 0.5 else "table_b"
        xy = _spot_on(table, rng)
        if np.hypot(*(xy - PLACE_MARK)) < MARK_CLEARANCE:
            continue
        position = np.array([*xy, TABLE_TOP + SHAPES[kind][1]])
        if _free(position, objects):
            objects.append(SceneObject(f"obj_{len(objects)}", kind, color, position))
    return Scene(seed, objects)


def _body(obj):
    radius, half = SHAPES[obj.kind]
    r, g, b = COLORS[obj.color]
    x, y, z = obj.position
    rgba = f"{r} {g} {b} 1"
    if obj.kind == "box":
        geoms = f'<geom name="{obj.name}_body" type="box" size="{radius} {radius} {half}" mass="0.3" rgba="{rgba}" friction="1.0"/>'
    else:
        geoms = f'<geom name="{obj.name}_body" type="cylinder" size="{radius} {half}" mass="0.33" rgba="{rgba}" friction="1.0"/>'
        if obj.kind == "bottle":
            geoms += f'\n      <geom name="{obj.name}_cap" type="cylinder" size="0.014 0.015" pos="0 0 {half + 0.015}" mass="0.02" material="cap"/>'
    return (
        f'    <body name="{obj.name}" pos="{x:.4f} {y:.4f} {z:.4f}">\n'
        f'      <freejoint name="{obj.name}"/>\n'
        f"      {geoms}\n"
        f"    </body>\n"
    )


def scene_xml(scene, template):
    """The MJCF text: the template room with its bottle moved and the other objects added."""
    bottle_block = re.search(r'    <!-- The bottle.*?</body>\n', template, re.S)
    if bottle_block is None:
        raise ValueError("robot/scene.xml no longer has the bottle block this generator replaces")
    target = scene.objects[0]
    bottle = re.sub(
        r'<body name="bottle" pos="[^"]*"',
        f'<body name="bottle" pos="{target.position[0]:.4f} {target.position[1]:.4f} {target.position[2]:.4f}"',
        bottle_block.group(0),
    )
    others = "".join(_body(o) for o in scene.objects[1:])
    return template.replace(bottle_block.group(0), bottle + others)


def write_scene(scene, robot_dir):
    """Write the scene next to the robot model, return its path."""
    robot_dir = Path(robot_dir)
    template = (robot_dir / "scene.xml").read_text()
    path = robot_dir / f"scene_seed{scene.seed}.xml"
    path.write_text(scene_xml(scene, template))
    return path


def write_live(robot_dir, seed=None):
    """robot/scene_live.xml, the room the dora dataflow runs in: scene.xml, or random room `seed`."""
    robot_dir = Path(robot_dir)
    template = (robot_dir / "scene.xml").read_text()
    text = template if seed is None else scene_xml(make_scene(seed), template)
    path = robot_dir / "scene_live.xml"
    path.write_text(text)
    return path


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "" else None
    path = write_live(Path(__file__).resolve().parents[1] / "robot", seed)
    if seed is None:
        print(f"{path.name}: the original room, one bottle")
    else:
        print(f"{path.name}: random room {seed}: " + ", ".join(o.label for o in make_scene(seed).objects))
