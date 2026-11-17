"""The world in MuJoCo: the G1 with its motor drivers and sensors, in any scene.

On the real robot, each motor has a small computer (the motor driver) that
gets a command (target angle, kp, kd, extra torque) and turns it into torque
thousands of times per second. Here the simulator plays that part, once per
2 ms physics step. The rest of the software sends the same commands it
would send to the real G1.
"""

import mujoco
import numpy as np

from g1 import robot


def yaw_of(quat):
    """Heading angle (rad) from an orientation quaternion (w, x, y, z)."""
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Simulator:
    def __init__(self, scene_path, start=(0.0, 0.0, 0.0)):
        """start: where the robot stands at the beginning, (x, y, heading)."""
        self.model = self.load_model(scene_path)
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(name).id for name in robot.JOINTS]
        self.qpos_index = self.model.jnt_qposadr[joints]
        self.dof_index = self.model.jnt_dofadr[joints]
        self.motor_index = np.array(
            [self.model.actuator(name).id for name in robot.JOINTS]
        )
        self.torque_limit = self.model.actuator_ctrlrange[self.motor_index, 1]
        self.gyro = self.model.sensor_adr[
            self.model.sensor("imu-pelvis-angular-velocity").id
        ]
        # the bottle is optional: scenes without one just have no bottle_pose
        bottle = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bottle")
        self.bottle_qpos = self.model.jnt_qposadr[bottle] if bottle >= 0 else None

        # motor commands, one entry per joint (zero gains = motor off)
        self.target = np.zeros(robot.NUM_JOINTS)
        self.kp = np.zeros(robot.NUM_JOINTS)
        self.kd = np.zeros(robot.NUM_JOINTS)
        self.tau = np.zeros(robot.NUM_JOINTS)
        self._stand_at(*start)

    def load_model(self, scene_path):
        """The MuJoCo model of the scene. Other worlds override this to add things."""
        return mujoco.MjModel.from_xml_path(scene_path)

    def _stand_at(self, x, y, heading):
        """Put the robot in its standing pose, feet on the floor, motors holding it."""
        pose = np.zeros(robot.NUM_JOINTS)
        pose[robot.LEGS] = robot.STAND_LEGS
        pose[robot.LEFT_ARM] = robot.LEFT_ARM_WALK
        pose[robot.LEFT_HAND] = robot.LEFT_HAND_OPEN
        pose[robot.RIGHT_ARM] = robot.RIGHT_ARM_WALK
        pose[robot.RIGHT_HAND] = robot.RIGHT_HAND_OPEN
        self.data.qpos[0:7] = [
            x,
            y,
            1.0,
            np.cos(heading / 2),
            0,
            0,
            np.sin(heading / 2),
        ]
        self.data.qpos[self.qpos_index] = pose
        mujoco.mj_kinematics(self.model, self.data)
        # lower the robot until its lowest foot point touches the floor
        # (each foot touches the ground with four small spheres)
        feet = [
            g
            for g in range(self.model.ngeom)
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            and "ankle_roll" in self.model.body(self.model.geom_bodyid[g]).name
        ]
        lowest = min(
            self.data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in feet
        )
        self.data.qpos[2] -= lowest
        mujoco.mj_forward(self.model, self.data)
        # until the controllers send commands, the motors hold this pose
        kp = np.concatenate(
            [
                robot.STAND_LEG_KP,
                robot.WAIST_KP,
                robot.ARM_KP,
                robot.HAND_KP,
                robot.ARM_KP,
                robot.HAND_KP,
            ]
        )
        kd = np.concatenate(
            [
                robot.STAND_LEG_KD,
                robot.WAIST_KD,
                robot.ARM_KD,
                robot.HAND_KD,
                robot.ARM_KD,
                robot.HAND_KD,
            ]
        )
        self.set_command(slice(0, robot.NUM_JOINTS), pose, kp, kd)

    # ---- commands in ----

    def set_command(self, group, target, kp, kd, tau=None):
        """New motor commands for a group of joints (a slice of robot.JOINTS)."""
        self.target[group] = target
        self.kp[group] = kp
        self.kd[group] = kd
        self.tau[group] = 0.0 if tau is None else tau

    # ---- physics ----

    def step(self):
        """2 ms of physics, with the motor drivers' torque law applied first."""
        q = self.data.qpos[self.qpos_index]
        dq = self.data.qvel[self.dof_index]
        torque = self.kp * (self.target - q) - self.kd * dq + self.tau
        self.data.ctrl[self.motor_index] = np.clip(
            torque, -self.torque_limit, self.torque_limit
        )
        mujoco.mj_step(self.model, self.data)

    # ---- sensors out ----

    @property
    def time(self):
        return self.data.time

    def joint_state(self):
        """Angles (rad) and speeds (rad/s) of the 43 joints, in robot.JOINTS order."""
        return self.data.qpos[self.qpos_index].copy(), self.data.qvel[
            self.dof_index
        ].copy()

    def imu(self):
        """Pelvis orientation (w, x, y, z) and turn rates (rad/s), like the G1's IMU."""
        return self.data.qpos[3:7].copy(), self.data.sensordata[
            self.gyro : self.gyro + 3
        ].copy()

    def base_pose(self):
        """Pelvis position (x, y, z) and orientation (w, x, y, z) in the world.

        On the real robot this comes from localization (lidar + map). Here we
        read it from the simulator.
        """
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()

    def world_state(self):
        """Every position in the scene (MuJoCo qpos): robot pose, joints, free objects."""
        return self.data.qpos.copy()

    def bottle_pose(self):
        """Bottle position and orientation. On the real robot: camera + detector."""
        return self.data.qpos[
            self.bottle_qpos : self.bottle_qpos + 3
        ].copy(), self.data.qpos[self.bottle_qpos + 3 : self.bottle_qpos + 7].copy()
