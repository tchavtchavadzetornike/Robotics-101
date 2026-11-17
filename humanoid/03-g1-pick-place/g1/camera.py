"""The head camera: draws what the robot sees, in color and depth.

On the real G1 this is the RealSense D435i in its face. Here a second copy of
the MuJoCo world is put into the same state as the simulation (world_state),
then drawn from the camera named in the model (robot/g1_with_hands.xml).

The camera follows the pinhole model. Four numbers, the intrinsics, say how a
point in front of the camera lands on a pixel:

    u = fx * X / Z + cx        X right, Y down, Z ahead (the image convention)
    v = fy * Y / Z + cy

MuJoCo describes the camera by its vertical field of view (fovy). Half the
image height sits at half the angle, so fy = (height / 2) / tan(fovy / 2).
Pixels are square, so fx = fy, and the image center is (width / 2, height / 2).

Two frames for the same camera:
  image  (OpenCV, Rerun):  x right, y down, z ahead
  MuJoCo:                  x right, y up,   looks along -z
They differ by turning half a turn around x: flip y and z. IMAGE_TO_MUJOCO does that.
"""

import math

import mujoco
import numpy as np

IMAGE_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def intrinsics(fovy_deg, width, height):
    """(fx, fy, cx, cy) in pixels, from the vertical field of view."""
    fy = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return fy, fy, width / 2, height / 2


class Camera:
    def __init__(self, scene_path, name, width, height):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.name = name
        self.id = self.model.camera(name).id
        self.width, self.height = width, height
        self.fx, self.fy, self.cx, self.cy = intrinsics(
            self.model.cam_fovy[self.id], width, height
        )
        self.renderer = mujoco.Renderer(self.model, height, width)

    def set_state(self, qpos):
        """Put every body where the simulation has it, and work out where things are."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def color(self):
        """(height, width, 3) uint8 RGB."""
        self.renderer.update_scene(self.data, camera=self.id)
        return self.renderer.render()

    def depth(self):
        """(height, width) float32: distance along the viewing axis (Z), in meters."""
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.id)
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth

    def segmentation(self):
        """(height, width) int32: the body id seen at each pixel, -1 for none.

        Only a simulator can give this: the exact truth of which object every
        pixel belongs to. Used to score detectors, never to drive the robot.
        """
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.id)
        ids = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        geom, kind = ids[..., 0], ids[..., 1]
        is_geom = (kind == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom >= 0)
        body = np.full(geom.shape, -1, dtype=np.int32)
        body[is_geom] = self.model.geom_bodyid[geom[is_geom]]
        return body

    def pose(self):
        """Camera position and rotation in the world, image convention (x right, y down, z ahead).

        rotation @ p_image + position turns a point seen by the camera into the world frame.
        """
        rotation = self.data.cam_xmat[self.id].reshape(3, 3) @ IMAGE_TO_MUJOCO
        return self.data.cam_xpos[self.id].copy(), rotation
