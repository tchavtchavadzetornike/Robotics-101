"""Run the whole mission without dora and without a window, and check the result.

The same parts as the dora graph (Simulator, Locomotion, ArmController,
Mission), wired together in one loop instead of exchanging messages. It runs
about 20 times faster than real time (about 2 s), so you can check every change.

  uv run python tests/test_mission.py           (prints the steps, PASS or FAIL)
  uv run python tests/test_mission.py --vision  (the bottle position from the head camera)
  uv run --with pytest pytest tests/            (both, as pytest tests)

With --vision the head camera renders 15 times per second of robot time and
g1/perception.py finds the bottle, as the camera and perception nodes do.
The mission never sees the simulator's bottle position then.
"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

LESSON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LESSON))

from g1 import messages, perception, robot  # noqa: E402
from g1.arm import ArmController  # noqa: E402
from g1.camera import Camera  # noqa: E402
from g1.locomotion import POLICY_DT, Locomotion  # noqa: E402
from g1.mission import BOTTLE_HALF_HEIGHT, Mission  # noqa: E402
from g1.simulator import Simulator  # noqa: E402

# The room, as in robot/scene.xml and dataflow.yml
PLACE_POINT = (1.0, -1.52)
PLACE_HEADING = math.radians(-90)
PICK_HEADING = 0.0
TABLE_HEIGHT = 0.80
TIME_LIMIT = 150.0  # s of robot time
CAMERA_DT = 1 / 15  # s: the head camera's frame time


def look(camera, sim, last):
    """One camera frame: where the bottle is, or `last` if it can't be seen now."""
    camera.set_state(sim.world_state())
    position, rotation = camera.pose()
    seen, _ = perception.find_bottle(
        camera.color(),
        camera.depth(),
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        position,
        rotation,
        0.03,
        BOTTLE_HALF_HEIGHT,
    )
    if seen is None:
        return last
    seen = perception.keep_height(seen, last)
    return last if seen is None else seen


def run_mission(verbose=True, vision=False, bottle_shift=(0.0, 0.0)):
    """bottle_shift: move the bottle on table A by (x, y) meters before starting."""
    sim = Simulator(str(LESSON / "robot/scene.xml"))
    sim.data.qpos[sim.bottle_qpos : sim.bottle_qpos + 2] += bottle_shift
    mujoco.mj_forward(sim.model, sim.data)
    camera = Camera(str(LESSON / "robot/scene.xml"), "head_cam", 640, 480) if vision else None
    seen_bottle = None
    next_frame = 0.0
    locomotion = Locomotion(str(LESSON / "policies/g1_walk.onnx"))
    arm = ArmController(str(LESSON / "robot/g1_with_hands.xml"))
    mission = Mission(PICK_HEADING, PLACE_POINT, PLACE_HEADING, TABLE_HEIGHT)

    arm_status = None
    arm_command_id = None
    label = ""
    next_control = 0.0
    lowest_pelvis = 1.0
    while sim.time < TIME_LIMIT and not mission.finished and not mission.failed:
        if vision and sim.time >= next_frame - 1e-9:  # the 15 Hz part: the camera
            next_frame += CAMERA_DT
            seen_bottle = look(camera, sim, seen_bottle)
        if sim.time >= next_control - 1e-9:  # the 50 Hz part: task, legs, arm
            next_control += POLICY_DT
            t = sim.time
            q, dq = sim.joint_state()
            quat, gyro = sim.imu()
            base_position, base_quat = sim.base_pose()
            bottle_position = seen_bottle if vision else sim.bottle_pose()[0]

            cmd_vel, command = mission.update(
                t,
                base_position,
                base_quat,
                bottle_position,
                locomotion.is_standing(t),
                arm_status,
            )
            if mission.label != label:
                label = mission.label
                if verbose and label:
                    print(f"t={t:6.2f}s  {label}")
            if command is not None:
                arm_command_id, kind, position, yaw = messages.unpack_arm_command(
                    command
                )
                if not arm.execute(t, kind, position, yaw):
                    arm_status = (arm_command_id, False)
                    arm_command_id = None

            sim.set_command(
                robot.LOWER_BODY,
                *locomotion.update(
                    t, cmd_vel, q[robot.LEGS], dq[robot.LEGS], quat, gyro
                ),
            )
            sim.set_command(robot.UPPER_BODY, *arm.update(t, q[robot.UPPER_BODY]))
            if arm_command_id is not None and not arm.busy():
                arm_status = (arm_command_id, True)
                arm_command_id = None
        sim.step()
        lowest_pelvis = min(lowest_pelvis, sim.data.qpos[2])
        if lowest_pelvis < 0.5:
            mission.failed = "the robot fell"

    bottle, bottle_quat = sim.bottle_pose()
    # tilt: angle between the bottle's axis and vertical (from its quaternion)
    w, x, y, z = bottle_quat
    tilt = math.degrees(math.acos(min(1.0, 1 - 2 * (x * x + y * y))))
    miss = float(np.hypot(bottle[0] - PLACE_POINT[0], bottle[1] - PLACE_POINT[1]))
    on_table = abs(bottle[2] - (TABLE_HEIGHT + 0.075)) < 0.01
    result = {
        "finished": mission.finished,
        "failed": mission.failed,
        "time": sim.time,
        "miss_cm": 100 * miss,
        "tilt_deg": tilt,
        "on_table": on_table,
    }
    if verbose:
        outcome = "finished" if mission.finished else f"FAILED: {mission.failed}"
        where = "standing on table B" if on_table else "NOT on table B"
        print(f"\nmission {outcome} at t={sim.time:.1f}s")
        print(
            f"bottle: {100 * miss:.1f} cm from the mark, tilt {tilt:.1f} deg, {where}"
        )
    return result


def test_mission():
    r = run_mission(verbose=False)
    assert r["finished"], r["failed"]
    assert r["on_table"] and r["miss_cm"] < 3.0 and r["tilt_deg"] < 10.0, r


def test_mission_vision():
    r = run_mission(verbose=False, vision=True)
    assert r["finished"], r["failed"]
    assert r["on_table"] and r["miss_cm"] < 3.0 and r["tilt_deg"] < 10.0, r


if __name__ == "__main__":
    r = run_mission(vision="--vision" in sys.argv)
    ok = r["finished"] and r["on_table"] and r["miss_cm"] < 3.0 and r["tilt_deg"] < 10.0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
