"""What the robot does, live in Rerun: a 3D map of the room, charts, the mission log.

It only listens, it sends nothing. You can remove it from the dataflow and the
robot does exactly the same thing.

Inputs:  base_pose     where the robot is and how far it leans, 50 Hz (sim)
         bottle_pose   where the bottle really is, 50 Hz (sim)
         seen_bottle   where the camera says it is, 10 Hz (perception)
         objects       the memory: every object seen, with its id (world)
         cmd_vel       the walking command (task)
         standing      1 = legs stand still and steady (locomotion)
         arm_command   the hand target, in the body frame (task)
         arm_status    which arm command finished (arm)
         step          the mission step as text (task), or the skill step (skills)
         conversation  what was heard, each tool call and result, each reply (agent, phase 5)
         rgb, depth    the head camera images, 10 Hz (camera)
         cam_pose      where the head camera is, 10 Hz (camera)
         tick          draw timer

The head camera is drawn twice: as a 2D image, and in the 3D room as a
pyramid (its field of view) with the image on it. Images are logged 2 times
a second (IMAGE_EVERY) and compressed: less work for the viewer, a small saved run.

The tables and the green mark come from robot/scene.xml, so the map always
matches the simulation.

VIEWER=rerun (default) opens the Rerun window, VIEWER=save writes out/g1.rrd.
"""

import json
import os

import mujoco
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from dora import Node

from g1 import frames, messages
from g1.simulator import yaw_of

TRAIL_POINTS = 2000  # how much of the walked path to keep (50 Hz: 40 s)
IMAGE_EVERY = float(os.environ.get("IMAGE_EVERY", "0.49"))  # s of robot time between logged images: every 5th frame of 10 Hz (2 Hz)
HAND_KINDS = {
    messages.REST: "arm down",
    messages.MOVE_JOINTS: "joint move",
    messages.MOVE_LINE: "line move",
    messages.OPEN: "open hand",
    messages.CLOSE: "close hand",
}
WOOD = (140, 97, 56)
WHITE = (225, 225, 220)
GREEN = (50, 200, 80)
BLUE = (50, 140, 230)
SEEN = (255, 210, 40)
OBJECT_COLORS = {"red": (220, 50, 50), "yellow": (240, 200, 30), "green": (60, 180, 80), "blue": (60, 130, 230)}
HALF_SIZES = {"bottle": (0.03, 0.03, 0.075), "can": (0.033, 0.033, 0.06), "box": (0.03, 0.03, 0.05)}
ROBOT = (240, 150, 60)
HAND = (230, 70, 160)


def start_rerun():
    if os.environ.get("VIEWER", "rerun") == "save":
        os.makedirs("out", exist_ok=True)
        rr.init("g1-dashboard")
        rr.save("out/g1.rrd")
    else:
        rr.init("g1-dashboard", spawn=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(name="room", origin="world"),
                    rrb.Horizontal(
                        rrb.Spatial2DView(name="head camera", origin="world/head_cam"),
                        rrb.Spatial2DView(name="depth", origin="head_depth"),
                    ),
                    row_shares=[3, 2],
                ),
                rrb.Vertical(
                    rrb.TextLogView(name="mission", origin="log"),
                    rrb.TextLogView(name="conversation", origin="talk"),
                    rrb.TextDocumentView(name="memory", origin="memory"),
                    rrb.TimeSeriesView(name="walk command", origin="cmd_vel"),
                    rrb.TimeSeriesView(name="balance", origin="balance"),
                    rrb.TimeSeriesView(name="camera error", origin="perception"),
                    rrb.TimeSeriesView(name="standing", origin="standing"),
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )
    )


def log_room(scene_path):
    """Tables and the place mark, read from the MuJoCo scene. Logged once."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for name, color in (("table_a_top", WOOD), ("table_b_top", WHITE)):
        geom = model.geom(name).id
        rr.log(
            f"world/{name}",
            rr.Boxes3D(
                centers=[data.geom_xpos[geom]],
                half_sizes=[model.geom_size[geom]],
                colors=[color],
                fill_mode="solid",
            ),
            static=True,
        )
    mark = data.site("place_target").xpos
    rr.log(
        "world/place_mark",
        rr.Points3D([mark], radii=[0.04], colors=[GREEN], labels=["mark"]),
        static=True,
    )


start_rerun()
log_room(os.environ["SCENE"])
node = Node()

robot_position = None
robot_heading = 0.0
trail = []
time = 0.0
true_bottle = None
rgb = depth = None  # newest images, with their metadata
last_image = -1e9
pinhole_logged = False

for event in node:
    if event["type"] == "STOP":
        break
    if event["type"] != "INPUT":
        continue
    name = event["id"]

    if name == "base_pose":
        time, robot_position, quat = messages.unpack_pose(event["value"].to_numpy())
        robot_heading = yaw_of(quat)
        rr.set_time("sim_time", duration=time)
        rr.log("balance/tilt_deg", rr.Scalars(np.degrees(frames.tilt_of(quat))))
        trail.append(robot_position.copy())
        del trail[:-TRAIL_POINTS]

    elif name == "bottle_pose":
        _, bottle, _ = messages.unpack_pose(event["value"].to_numpy())
        true_bottle = bottle
        rr.set_time("sim_time", duration=time)
        rr.log(
            "world/bottle",
            rr.Boxes3D(
                centers=[bottle], half_sizes=[[0.03, 0.03, 0.075]], colors=[BLUE]
            ),
        )

    elif name == "seen_bottle":
        seen_time, seen, _ = messages.unpack_pose(event["value"].to_numpy())
        rr.set_time("sim_time", duration=seen_time)
        rr.log(
            "world/bottle_seen",
            rr.Boxes3D(
                centers=[seen],
                half_sizes=[[0.03, 0.03, 0.075]],
                colors=[SEEN],
                labels=["seen"],
            ),
        )
        if true_bottle is not None:
            error = 100 * (seen - true_bottle)
            rr.log("perception/xy_error_cm", rr.Scalars(np.hypot(error[0], error[1])))
            rr.log("perception/z_error_cm", rr.Scalars(error[2]))

    elif name == "objects":
        remembered = json.loads(event["value"][0].as_py())
        rr.set_time("sim_time", duration=event["metadata"]["time"])
        rr.log(
            "world/memory",
            rr.Boxes3D(
                centers=[o["position"] for o in remembered],
                half_sizes=[HALF_SIZES.get(o["kind"], (0.03, 0.03, 0.06)) for o in remembered],
                colors=[OBJECT_COLORS.get(o["color"], (200, 200, 200)) for o in remembered],
                labels=[f"#{o['id']} {o['color']} {o['kind'] or 'object'}" for o in remembered],
            ),
        )
        lines = [
            f"#{o['id']:<3} {o['color']:<7} {o['kind'] or '?':<7} at ({o['position'][0]:.2f}, {o['position'][1]:.2f})"
            f"  seen {o['sightings']}x, last t={o['last_seen']:.1f}s"
            for o in remembered
        ]
        rr.log("memory", rr.TextDocument("\n".join(lines) or "nothing seen yet"))

    elif name == "cmd_vel":
        forward, left, turn = event["value"].to_numpy()
        rr.set_time("sim_time", duration=time)
        rr.log("cmd_vel/forward_m_s", rr.Scalars(forward))
        rr.log("cmd_vel/left_m_s", rr.Scalars(left))
        rr.log("cmd_vel/turn_rad_s", rr.Scalars(turn))

    elif name == "standing":
        rr.set_time("sim_time", duration=time)
        rr.log("standing/steady", rr.Scalars(event["value"].to_numpy()[0]))

    elif name == "conversation":
        rr.set_time("sim_time", duration=time)
        rr.log("talk", rr.TextLog(event["value"][0].as_py(), level="INFO"))

    elif name == "step":
        rr.set_time("sim_time", duration=time)
        rr.log("log", rr.TextLog(event["value"][0].as_py(), level="INFO"))

    elif name == "arm_command" and robot_position is not None:
        command_id, kind, target, _ = messages.unpack_arm_command(
            event["value"].to_numpy()
        )
        rr.set_time("sim_time", duration=time)
        rr.log(
            "log",
            rr.TextLog(f"arm #{command_id}: {HAND_KINDS[kind]}", level="DEBUG"),
        )
        if kind in (messages.MOVE_JOINTS, messages.MOVE_LINE):
            world = frames.to_world(target, robot_position, robot_heading)
            rr.log(
                "world/hand_target",
                rr.Points3D([world], radii=[0.02], colors=[HAND], labels=["hand"]),
            )

    elif name == "arm_status":
        command_id, ok = messages.unpack_arm_status(event["value"].to_numpy())
        rr.set_time("sim_time", duration=time)
        level = "DEBUG" if ok else "ERROR"
        rr.log("log", rr.TextLog(f"arm #{command_id}: {'done' if ok else 'failed'}", level=level))

    elif name == "rgb":
        rgb = (event["value"].to_numpy(), event["metadata"])

    elif name == "depth":
        depth = (event["value"].to_numpy(), event["metadata"])

    elif name == "cam_pose":
        # The camera sends rgb, depth, then cam_pose: when the pose arrives, the
        # images of the same moment are already here.
        pose = event["value"].to_numpy()
        image_time = pose[0]
        if rgb is None or rgb[1]["time"] != image_time:
            continue
        if image_time - last_image < IMAGE_EVERY:
            continue
        last_image = image_time
        meta = rgb[1]
        width, height = meta["width"], meta["height"]
        rr.set_time("sim_time", duration=image_time)
        if not pinhole_logged:
            rr.log(
                "world/head_cam",
                rr.Pinhole(
                    focal_length=[meta["fx"], meta["fy"]],
                    principal_point=[meta["cx"], meta["cy"]],
                    width=width,
                    height=height,
                    camera_xyz=rr.ViewCoordinates.RDF,  # x right, y down, z ahead
                    image_plane_distance=0.4,
                ),
                static=True,
            )
            pinhole_logged = True
        rr.log(
            "world/head_cam",
            rr.Transform3D(translation=pose[1:4], mat3x3=pose[4:13].reshape(3, 3)),
        )
        image = rgb[0].reshape(height, width, 3)
        rr.log("world/head_cam", rr.Image(image).compress(jpeg_quality=75))
        if depth is not None and depth[1]["time"] == image_time:
            meters = depth[0].reshape(height, width)[::2, ::2]  # half size is enough to look at
            millimeters = np.clip(meters * 1000, 0, 65535).astype(np.uint16)  # like the RealSense
            rr.log("head_depth", rr.DepthImage(millimeters, meter=1000.0))

    elif name == "tick" and robot_position is not None:
        rr.set_time("sim_time", duration=time)
        ahead = 0.3 * np.array([np.cos(robot_heading), np.sin(robot_heading), 0.0])
        rr.log(
            "world/robot",
            rr.Arrows3D(origins=[robot_position], vectors=[ahead], colors=[ROBOT]),
        )
        rr.log(
            "world/robot_path",
            rr.LineStrips3D([np.array(trail)], colors=[ROBOT], radii=[0.005]),
        )
