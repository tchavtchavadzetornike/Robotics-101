# The robot and the room (MJCF files)

| File | What it is |
| --- | --- |
| `g1_with_hands.xml` | the Unitree G1: 29 body joints + two Dex3-1 hands (7 joints each) = 43 motors, 34.4 kg |
| `assets/` | the 3D shapes (meshes) of every part |
| `scene.xml` | the room: floor, light, table A with the bottle, table B with the green mark; includes the robot |
| `LICENSE` | BSD-3-Clause, for `g1_with_hands.xml` and `assets/` |

## Where the G1 comes from

[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1), the collection of robot models curated by the MuJoCo team, commit `98a70c5` (2026-11-02). They made it from Unitree's own description (`unitree_ros`, `g1_29dof_with_hand_rev_1_0`).

## What we changed in `g1_with_hands.xml`

1. **Motors take torque.** Menagerie gives every joint a position servo with one stiffness (kp = 500, marked "needs tuning"). The real G1 works differently: each motor driver takes a target angle, kp, kd and an extra torque, and computes the torque itself. So the actuators are now plain torque motors (`<motor>`, limit = the motor's real peak torque), and `g1/simulator.py` does the driver's job.
2. **Removed the "stand" keyframe.** It held servo targets, which mean nothing for torque motors. `g1/simulator.py` sets up the start pose instead.
3. **Added two sites** `left_hand_grasp` and `right_hand_grasp`: the point in the middle of each hand where a 6 cm bottle sits when the fingers close. The arm's inverse kinematics moves this point.
4. **Added a head camera** `head_cam` in `torso_link`, where the real G1 has its RealSense D435i: 8 cm ahead, 42 cm up, tilted 40° down, 58° vertical field of view. It sees the bottle both from the start position and while grasping.

Nothing else (masses, shapes, joint limits) is changed.
