"""Skills: what the robot can do, as reusable building blocks with honest results.

    look_around()          survey both tables, the memory fills up
    go_to(place)           "table_a", "table_b", "mark", "start"
    pick(object_id)        walk to the best side, grasp, lift, check the hand isn't empty
    place(target)          "table_a", "table_b", "mark", or ("next_to", object_id)

A skill is a list of steps, like mission.py, but each step computes its
target from what the robot knows *when the step starts*: where it stands now,
where the memory says the object is now. Every skill ends with a Result: ok,
or failed with a reason a person (and in phase 5 a language model) can read.

Checks before (can it be done at all?) come from reach.py: reachable,
nothing in the hand's way, a free spot. Checks during: the hand closed on
something (finger angles, GRIP_EMPTY), the object is still where it was.
Check after: the object stands where it was put.

If a step fails while the arm is near the table, the skill retreats before it
reports: open the hand, hand up, carry pose, step back, arm down. Lowering the
arm at the table is what nearly toppled the robot in phase 0.

The steps:
  ("walk", pose_fn)          walk there, around tables (paths.py)
  ("look", seconds)          stand still and let the detector look
  ("still",)                 wait until the legs stand still (first step of every retreat)
  ("arm", kind, target_fn)   one arm command (messages.py), target in the body frame
  ("check", fn)              fn(robot) -> None, or the reason to fail
  ("do", fn)                 fn(robot): bookkeeping, e.g. "now holding #3"
"""

from dataclasses import dataclass, field

import numpy as np

from g1 import frames, messages, navigation, paths, reach, robot, scenes, survey
from g1.mission import APPROACH, CARRY, GRASP_SPOT, LIFT, READY

GRIP_EMPTY = 0.4  # rad: closed fingers this near the closed pose hold nothing (measured: 0.03-0.18 empty, 0.74 holding)
APPROACH_BACK = 0.45  # m: the approach point is this far back from a stand pose
WAYPOINT_PASS = 0.15  # m: close enough to a waypoint on the way to go on to the next
REACH_SLACK = 0.10  # m: the spot may be this far from GRASP_SPOT, seen from where the robot really stands
STALL_NEAR = 0.15  # m: this close to a goal...
STALL_TIME = 3.0  # s: ...and not 1 cm further in this long: the walking policy ignores the small command, count as there
PLACE_GAP = 0.01  # m: let go this far above the table
TUCK = np.array([0.15, -0.18, 0.18])  # m, body frame: the carry pose 5 cm nearer...
TUCK_YAW = np.pi / 2  # ...with the fingers turned in, along the table edge: for stepping back with an EMPTY hand.
# Carrying an object stays in CARRY: with the object tucked in, the robot drifted 7-12 cm and fell
# twice in 10 rooms (tell-the-robot phase 4); the walking policy wants the arms near its training pose.
PULL_BACK = 0.10  # m: after letting go, the hand slides straight back along the object's side
ATTEMPTS = 3  # places and grasps try up to this many options (spots, table sides) before giving up
NEIGHBORHOOD = 0.3  # m: objects this close to a put-down spot are checked for being pushed
DISTURBED = 0.03  # m: a neighbor that moved more than this was pushed
STOP_WAIT = 0.5  # s
WALK_TIMEOUT = 90.0  # s
ARM_TIMEOUT = 20.0  # s
PLACES = {
    "start": (0.0, 0.0, 0.0),
    "table_a": (0.95, 0.0, 0.0),
    "table_b": (1.0, -1.05, -np.pi / 2),
    "mark": (1.0, -1.05, -np.pi / 2),
}


@dataclass
class Result:
    ok: bool
    message: str


@dataclass
class Robot:
    """What a skill knows at one moment. The skills node (or a test) fills it in every 20 ms."""

    time: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # pelvis, world
    heading: float = 0.0
    standing: bool = False
    arm_status: tuple | None = None  # (command id, ok) of the last finished arm command
    hand_q: np.ndarray = field(default_factory=lambda: robot.RIGHT_HAND_OPEN.copy())
    objects: list = field(default_factory=list)  # the memory: [{id, kind, color, position, last_seen}]
    held: int | None = None  # id of the object in the hand
    released_at: np.ndarray | None = None  # set by place: where the held object now stands
    arm_area: np.ndarray | None = None  # (x, y) where the arm works at a table: the memory can't check there

    @property
    def pose(self):
        return (self.position[0], self.position[1], self.heading)

    def object(self, object_id):
        return next((o for o in self.objects if o["id"] == object_id), None)

    def others(self, but=None):
        return [np.array(o["position"]) for o in self.objects if o["id"] not in (but, self.held)]

    def to_body(self, point):
        return frames.to_body(point, self.position, self.heading)

    def grip_empty(self):
        return float(np.linalg.norm(self.hand_q - robot.RIGHT_HAND_CLOSED)) < GRIP_EMPTY


def approach_point(stand):
    """APPROACH_BACK behind a stand pose, facing the same way: outside the grown table."""
    x, y, heading = stand
    return np.array([x - APPROACH_BACK * np.cos(heading), y - APPROACH_BACK * np.sin(heading), heading])


def walk_route(start, goal):
    """Waypoints from start to goal pose, around tables. None if there is no way."""
    goal = np.asarray(goal, float)
    boxes = paths.obstacles()
    target = goal if not any(paths.inside(goal, b) for b in boxes) else approach_point(goal)
    if any(paths.inside(start[:2], b) for b in boxes):
        start = approach_point((start[0], start[1], start[2]))  # leave the table first, backwards
        first = [np.array([start[0], start[1], start[2]])]
    else:
        first = []
    route = paths.plan(start[:2], target[:2], boxes)
    if route is None:
        return None
    poses = list(first)
    for a, b in zip(route[:-1], route[1:-1] + [None]):
        if b is None:
            break
        poses.append(np.array([b[0], b[1], np.arctan2(b[1] - a[1], b[0] - a[0])]))
    poses.append(np.array([*target[:2], target[2]]))
    if target is not goal:
        poses[-1] = np.array([*poses[-1], 1.0])  # the approach point: arrive there, facing the table, don't cut it
        poses.append(goal)
    return poses


def in_reach(r, spot, what):
    """None if `spot` (x, y) is where the hand reaches well from where the robot really stands."""
    ahead, left, _ = r.to_body([*spot, 0.0])
    off = float(np.hypot(ahead - GRASP_SPOT[0], left - GRASP_SPOT[1]))
    if off > REACH_SLACK:
        return f"{what} is {100 * off:.0f} cm from where my hand reaches best from here (I stand {100 * off:.0f} cm off)"
    return None


def must_arrive(pose):
    """Waypoints marked by walk_route as places to stop at, not to pass by."""
    return len(pose) == 4


class Skill:
    name = "skill"

    def __init__(self):
        self.steps = []
        self.index = -1
        self.result = None
        self.retreating = False
        self.navigator = navigation.Navigator()
        self.step_start = 0.0
        self.route = []
        self.stopped_at = None
        self.command_id = 0
        self.hand = READY.copy()
        self.last_time = None
        self.label = ""
        self.progress_at, self.progress_time, self.stalled_here = None, 0.0, False
        self.lined_up_again = False

    # ---- what each skill defines ----

    def plan(self, r):
        """The steps, or a Result if it can't be done at all."""
        raise NotImplementedError

    def retreat_steps(self, r):
        return []

    # ---- running ----

    @property
    def done(self):
        return self.result is not None

    @property
    def looking(self):
        return 0 <= self.index < len(self.steps) and self.steps[self.index][0] == "look"

    def try_other(self, r, reason):
        """Steps for another way to do it after `reason`, or None. Default: none."""
        return None

    def failure_message(self, r, reason):
        return reason

    def fail(self, r, reason):
        if self.retreating:
            self.result = Result(False, self.failure_message(r, reason))
            return
        other = self.try_other(r, reason)
        if other is not None:
            self.steps, self.index = other, -1
            return
        steps = self.retreat_steps(r)
        if steps:
            steps = [("still", "stand still first")] + steps  # moving the arm while stepping toppled the robot
        self.failure = reason
        self.retreating = True
        self.steps, self.index = steps, -1
        if not steps:
            self.result = Result(False, self.failure_message(r, reason))

    def update(self, r):
        """Call every 20 ms. Returns (cmd_vel, arm_command or None)."""
        dt = 0.0 if self.last_time is None else r.time - self.last_time
        self.last_time = r.time
        stop = (0.0, 0.0, 0.0)
        if self.result is not None:
            return stop, None
        if self.index < 0 and not self.retreating:
            planned = self.plan(r)
            if isinstance(planned, Result):
                self.result = planned
                return stop, None
            self.steps = planned
        if self.index < 0:
            return self._next(r)
        step = self.steps[self.index]
        kind = step[0]

        if kind == "walk":
            if r.time - self.step_start > WALK_TIMEOUT:
                self.fail(r, f"{self.label}: still not there after {WALK_TIMEOUT:.0f} s")
                return stop, None
            while len(self.route) > 1 and not must_arrive(self.route[0]) and np.hypot(r.position[0] - self.route[0][0], r.position[1] - self.route[0][1]) < WAYPOINT_PASS:
                self.route.pop(0)  # close enough to a waypoint on the way: on to the next
                self.progress_at = None
            goal = self.route[0][:3]
            final = len(self.route) == 1
            if self.stopped_at is None and self._stalled(r, goal):
                if not final and not must_arrive(self.route[0]):
                    self.route.pop(0)
                    return stop, None
                off = np.hypot(r.position[0] - goal[0], r.position[1] - goal[1])
                if final and self._precise(step) and off > REACH_SLACK:
                    return self._line_up_again(r, step)
                self.stopped_at = r.time
            if self.stopped_at is None:
                cmd_vel, arrived = self.navigator.update(r.pose, goal, dt)
                if not arrived:
                    return cmd_vel, None
                if not final and not must_arrive(self.route[0]):
                    self.route.pop(0)
                    return stop, None
                self.stopped_at = r.time
            if not r.standing or r.time - self.stopped_at < STOP_WAIT:
                return stop, None
            if not navigation.close_to(r.pose, goal) and not self.stalled_here:
                self.stopped_at = None
                return stop, None
            if not final:  # stood still at the approach point, lined up: now straight in
                self.route.pop(0)
                self.stopped_at, self.progress_at, self.stalled_here = None, None, False
                self.navigator.stop()
                return stop, None
            return self._next(r)

        if kind == "look":
            if r.time - self.step_start >= step[1]:
                return self._next(r)
            return stop, None

        if kind == "still":
            if r.standing and r.time - self.step_start > STOP_WAIT:
                return self._next(r)
            return stop, None

        if kind == "arm":
            if r.time - self.step_start > ARM_TIMEOUT:
                self.fail(r, f"{self.label}: the arm took longer than {ARM_TIMEOUT:.0f} s")
                return stop, None
            if r.arm_status is not None and r.arm_status[0] == self.command_id:
                if not r.arm_status[1]:
                    self.fail(r, f"{self.label}: the hand can't reach {np.round(self.hand, 2).tolist()}")
                    return stop, None
                return self._next(r)
            return stop, None

        return self._next(r)

    @staticmethod
    def _precise(step):
        return len(step) == 4 and step[2] == "precise"

    def _line_up_again(self, r, step):
        """Stuck short of a stand pose: back out to the approach point and come in again, once."""
        stop = (0.0, 0.0, 0.0)
        if self.lined_up_again:
            self.fail(r, f"{self.label}: couldn't line up at the table, stuck {100 * np.hypot(r.position[0] - self.route[-1][0], r.position[1] - self.route[-1][1]):.0f} cm off")
            return stop, None
        self.lined_up_again = True
        goal = self.route[-1][:3]
        self.route = [np.array([*approach_point(goal), 1.0]), goal]
        self.stopped_at, self.progress_at, self.stalled_here = None, None, False
        self.navigator.stop()
        return stop, None

    def _stalled(self, r, goal):
        """Near the goal and not getting closer for STALL_TIME: the policy's dead zone.

        "Closer" is the distance to the goal, not movement: stepping in place sways the
        pelvis 1-2 cm, so it moves without getting anywhere.
        """
        distance = float(np.hypot(*(np.asarray(r.position[:2]) - np.asarray(goal[:2]))))
        if self.progress_at is None or distance < self.progress_at - 0.01:
            self.progress_at, self.progress_time = distance, r.time
            return False
        self.stalled_here = distance < STALL_NEAR and r.time - self.progress_time > STALL_TIME
        return self.stalled_here

    def _next(self, r):
        stop = (0.0, 0.0, 0.0)
        self.progress_at, self.stalled_here = None, False
        while True:
            self.index += 1
            self.step_start = r.time
            self.stopped_at = None
            self.navigator.stop()
            if self.index >= len(self.steps):
                r.arm_area = None
                if self.retreating:
                    self.result = Result(False, self.failure_message(r, self.failure))
                else:
                    self.result = self.success(r)
                return stop, None
            step = self.steps[self.index]
            kind = step[0]
            self.label = step[-1] if isinstance(step[-1], str) else kind
            if kind == "check":
                reason = step[1](r)
                if reason:
                    self.fail(r, reason)
                    if self.result is not None or not self.steps:
                        return stop, None
                continue
            if kind == "do":
                step[1](r)
                continue
            if kind == "walk":
                self.lined_up_again = False
                route = walk_route(r.pose, step[1](r))
                if route is None:
                    self.fail(r, f"{self.label}: no free path")
                    if self.result is not None:
                        return stop, None
                    continue
                self.route = route
                return stop, None
            if kind in ("look", "still"):
                return stop, None
            if kind == "arm":
                arm_kind, target = step[1], step[2]
                yaw = step[3] if len(step) == 5 else 0.0
                if target is not None:
                    self.hand = np.asarray(target(r), float)
                self.command_id += 1
                return stop, messages.pack_arm_command(self.command_id, arm_kind, self.hand, yaw)

    def success(self, r):
        return Result(True, "done")


class LookAround(Skill):
    name = "look_around"

    def plan(self, r):
        steps = []
        for kind, value in survey.look_around_plan():
            if kind == "walk":
                steps.append(("walk", lambda r, v=value: np.array(v), "walk to a viewpoint"))
            else:
                steps.append(("look", value, "look"))
        return steps

    def success(self, r):
        labels = ", ".join(f"#{o['id']} {o['color']} {o['kind'] or 'object'}" for o in r.objects)
        return Result(True, f"I see {len(r.objects)} objects: {labels}" if r.objects else "I see no objects")


class GoTo(Skill):
    name = "go_to"

    def __init__(self, place):
        super().__init__()
        self.place = place

    def plan(self, r):
        if self.place not in PLACES:
            return Result(False, f"unknown place {self.place!r}, I know {', '.join(PLACES)}")
        return [("walk", lambda r: np.array(PLACES[self.place]), f"walk to {self.place}")]

    def success(self, r):
        return Result(True, f"at {self.place}")


class Pick(Skill):
    name = "pick"

    def __init__(self, object_id):
        super().__init__()
        self.object_id = object_id

    def target(self, r):
        return np.array(r.object(self.object_id)["position"])

    def plan(self, r):
        if r.held is not None:
            return Result(False, f"my hand already holds #{r.held}; place it first")
        obj = r.object(self.object_id)
        if obj is None:
            return Result(False, f"I don't know object #{self.object_id}; look around first")
        self.what = f"#{obj['id']} {obj['color']} {obj['kind'] or 'object'}"
        if obj.get("standing") is False:
            return Result(False, f"can't pick {self.what}: it is lying on its side, the hand only grasps standing objects")
        options, reason = reach.grasp_options(obj["position"], r.others(but=self.object_id))
        if not options:
            return Result(False, f"can't pick {self.what}: {reason}")
        self.options = sorted(options, key=lambda o: self._route_length(r, o.stand))[:ATTEMPTS]
        self.attempt = 0
        return self.steps_for(r, self.options[0])

    def steps_for(self, r, option):
        """The steps to pick the object from one option (a table side and where to stand)."""
        self.option = option
        stand = option.stand
        return [
            ("walk", lambda r: stand, "precise", f"walk to {self.what}"),
            ("look", 1.0, "look at it up close"),
            ("check", self._still_there, "check it is still there"),
            ("check", lambda r: in_reach(r, self.target(r)[:2], self.what), "check it is in reach from here"),
            ("do", lambda r: setattr(r, "arm_area", self.option.spot), "arm at work"),
            ("arm", messages.MOVE_JOINTS, lambda r: READY.copy(), "raise the hand"),
            ("arm", messages.MOVE_LINE, lambda r: r.to_body(self.target(r)) + [0.0, -APPROACH, 0.0], "open hand next to it"),
            ("arm", messages.MOVE_LINE, lambda r: r.to_body(self.target(r)), "hand around it"),
            ("arm", messages.CLOSE, None, "close the hand"),
            ("check", self._holding, "check the hand holds it"),
            ("do", self._hold, "now holding it"),
            ("arm", messages.MOVE_LINE, lambda r: self.hand + [0.0, 0.0, LIFT], "lift it"),
            ("arm", messages.MOVE_JOINTS, lambda r: CARRY.copy(), "hold it close"),
            ("check", self._holding, "check it is still in the hand"),
            ("walk", lambda r: approach_point(r.pose), "step back"),
            ("do", lambda r: setattr(r, "arm_area", None), "arm done"),
        ]

    def try_other(self, r, reason):
        """Couldn't line up or reach from this side: back away and try the next side, if there is one."""
        if r.held is not None or not hasattr(self, "options"):
            return None
        if "line up" not in reason and "reach" not in reason:
            return None
        self.attempt += 1
        if self.attempt >= len(self.options):
            return None
        back = [("still", "stand still first"), ("walk", lambda r: approach_point(r.pose), "step back")]
        return back + self.steps_for(r, self.options[self.attempt])

    def _route_length(self, r, stand):
        route = walk_route(r.pose, stand)
        return np.inf if route is None else paths.length([np.asarray(r.position[:2])] + [p[:2] for p in route])

    def _still_there(self, r):
        obj = r.object(self.object_id)
        if obj is None:
            return f"{self.what} is gone: it was not there when I looked again"
        return None

    def _holding(self, r):
        return f"{self.what}: my hand closed on nothing" if r.grip_empty() else None

    def _hold(self, r):
        r.held = self.object_id

    def retreat_steps(self, r):
        if self.index < 3:
            return []
        stand = self.option.stand
        return [
            ("do", self._release, "let go"),
            ("arm", messages.OPEN, None, "open the hand"),
            ("arm", messages.MOVE_LINE, lambda r: self.hand + [0.0, -APPROACH, 0.05], "hand away"),
            ("arm", messages.MOVE_LINE, lambda r: READY.copy(), "raise the hand"),
            ("arm", messages.MOVE_JOINTS, lambda r: READY.copy(), TUCK_YAW, "turn the fingers in"),
            ("arm", messages.MOVE_LINE, lambda r: TUCK.copy(), TUCK_YAW, "tuck the hand"),
            ("walk", lambda r: approach_point(r.pose), "step back"),
            ("arm", messages.REST, None, "arm down"),
        ]

    def _release(self, r):
        r.held = None

    def success(self, r):
        return Result(True, f"holding {self.what}")


class Place(Skill):
    name = "place"

    def __init__(self, target):
        super().__init__()
        self.target = target  # "table_a" | "table_b" | "mark" | ("next_to", object_id)
        self.hand = CARRY.copy()  # where pick left the hand

    def plan(self, r):
        if r.held is None:
            return Result(False, "my hand is empty; pick something first")
        held = r.object(r.held)
        self.what = f"#{r.held} {held['color']} {held['kind'] or 'object'}" if held else f"#{r.held}"
        self.held_id = r.held
        self.half = scenes.SHAPES.get(held["kind"] if held else "bottle", scenes.SHAPES["bottle"])[1]
        target, where = self.target, self.target
        if isinstance(target, tuple):
            anchor = r.object(target[1])
            if anchor is None:
                return Result(False, f"I don't know object #{target[1]}")
            target = ("next_to", np.array(anchor["position"]))
            where = f"next to #{anchor['id']} {anchor['color']} {anchor['kind'] or 'object'}"
        self.where = where
        options, reason = reach.place_options(target, r.others(but=r.held), anchor=r.position)
        if not options:
            return Result(False, f"can't place {self.what}: {reason}")
        self.options = sorted(options[:8], key=lambda o: Pick._route_length(self, r, o.stand))[:ATTEMPTS]
        self.attempt = 0
        return [("check", self._in_hand, "check it is in the hand")] + self.steps_for(r, self.options[0])

    def steps_for(self, r, option):
        """The steps to put the held object down at one option (a spot and where to stand)."""
        self.option = option
        self.neighbors = {
            o["id"]: (np.array(o["position"]), o.get("standing", True))
            for o in r.objects
            if o["id"] != r.held and np.hypot(*(np.array(o["position"][:2]) - option.spot)) < NEIGHBORHOOD
        }
        stand, spot = option.stand, option.spot
        center_z = scenes.TABLE_TOP + self.half + PLACE_GAP
        return [
            ("walk", lambda r: stand, "precise", f"carry it to {self.where}"),
            ("check", self._in_hand, "check it is still in the hand"),
            ("check", lambda r: in_reach(r, spot, "the spot"), "check the spot is in reach from here"),
            ("do", lambda r: setattr(r, "arm_area", spot), "arm at work"),
            ("arm", messages.MOVE_LINE, lambda r: np.array([*r.to_body([*spot, 0.0])[:2], self.hand[2]]), "hold it over the spot"),
            ("arm", messages.MOVE_LINE, lambda r: r.to_body([*spot, center_z]), "put it down"),
            ("arm", messages.OPEN, None, "open the hand"),
            ("do", self._release, "let go"),
            ("arm", messages.MOVE_LINE, lambda r: self.hand + [-PULL_BACK, 0.0, 0.0], "pull the hand straight back"),
            ("arm", messages.MOVE_LINE, lambda r: READY.copy(), "raise the hand"),
            ("arm", messages.MOVE_JOINTS, lambda r: READY.copy(), TUCK_YAW, "turn the fingers in"),
            ("arm", messages.MOVE_LINE, lambda r: TUCK.copy(), TUCK_YAW, "tuck the hand"),
            ("walk", lambda r: approach_point(r.pose), "step back"),
            ("arm", messages.REST, None, "arm down"),
            ("do", lambda r: setattr(r, "arm_area", None), "arm done"),
            ("do", self._mark_look, "remember when I looked"),
            ("look", 2.5, "look where I put it"),
            ("check", self._there, "check it stands there"),
        ]

    def holding_retreat(self):
        """Back away from the table without letting go: hand back over the floor, fingers in, step back."""
        return [
            ("still", "stand still first"),
            ("arm", messages.MOVE_JOINTS, lambda r: CARRY.copy(), "hold it close"),
            ("walk", lambda r: approach_point(r.pose), "step back"),
            ("do", lambda r: setattr(r, "arm_area", None), "arm done"),
        ]

    def try_other(self, r, reason):
        if r.held is None or not hasattr(self, "options"):
            return None  # let go already, or fell out: nothing to retry
        if "line up" not in reason and "reach" not in reason:
            return None
        self.attempt += 1
        if self.attempt >= len(self.options):
            return None
        self.tried = getattr(self, "tried", []) + [reason]
        return self.holding_retreat() + self.steps_for(r, self.options[self.attempt])

    def failure_message(self, r, reason):
        if r.held is not None:
            tries = f" (tried {self.attempt + 1} spots)" if getattr(self, "attempt", 0) else ""
            return f"{reason}{tries}; I'm still holding {self.what}"
        return reason

    def _in_hand(self, r):
        if r.grip_empty():
            r.held = None
            return f"{self.what} fell out of my hand; I don't know where it is now, look around"
        return None

    def retreat_steps(self, r):
        if r.held is not None:  # it is still in the hand: keep it, just back away
            return self.holding_retreat()[1:]
        if self.index == 0:  # failed before walking anywhere: away from tables, nothing to bump into
            return [("arm", messages.OPEN, None, "open the hand"), ("arm", messages.REST, None, "arm down")]
        # anywhere later the robot may stand at a table: lowering the arm there hits the table (phase 0)
        return [
            ("arm", messages.OPEN, None, "open the hand"),
            ("arm", messages.MOVE_LINE, lambda r: self.hand + [-PULL_BACK, 0.0, 0.0], "pull the hand back"),
            ("arm", messages.MOVE_LINE, lambda r: READY.copy(), "raise the hand"),
            ("arm", messages.MOVE_JOINTS, lambda r: READY.copy(), TUCK_YAW, "turn the fingers in"),
            ("arm", messages.MOVE_LINE, lambda r: TUCK.copy(), TUCK_YAW, "tuck the hand"),
            ("walk", lambda r: approach_point(r.pose), "step back"),
            ("arm", messages.REST, None, "arm down"),
        ]

    def _release(self, r):
        r.held = None
        r.released_at = np.array([*self.option.spot, scenes.TABLE_TOP + self.half])

    def _mark_look(self, r):
        self.looked_from = r.time

    def _there(self, r):
        """Seen at the spot *after* letting go. The memory was told where it was put, so
        only a fresh sighting proves it (last_seen after the look started)."""
        lying = False
        for o in r.objects:
            fresh = o.get("last_seen", 0.0) >= self.looked_from
            near = np.hypot(*(np.array(o["position"][:2]) - self.option.spot)) < (0.05 if o.get("standing", True) else 0.12)
            if fresh and near:
                if o.get("standing", True):
                    return None
                lying = True
        if lying:
            return f"{self.what} fell over: it lies on its side where I put it down"
        return f"{self.what} is not where I put it: I don't see it there"

    def success(self, r):
        message = f"put {self.what} {self.where if isinstance(self.target, tuple) else 'on ' + self.where}"
        moved = []
        for object_id, (before, stood) in self.neighbors.items():
            now = r.object(object_id)
            if now is None:
                moved.append(f"#{object_id} is gone")
                continue
            if stood and now.get("standing", True) is False:
                moved.append(f"#{object_id} {now['color']} {now['kind'] or 'object'} fell over")
                continue
            if now.get("last_seen", 0.0) < self.looked_from:
                moved.append(f"I can't see #{object_id} {now['color']} {now['kind'] or 'object'} anymore, it may have been knocked off")
                continue
            shift = np.hypot(*(np.array(now["position"][:2]) - before[:2]))
            if shift > DISTURBED:
                moved.append(f"#{object_id} {now['color']} {now['kind'] or 'object'} moved {100 * shift:.0f} cm")
        if moved:
            message += ", but " + " and ".join(moved)
        return Result(True, message)
