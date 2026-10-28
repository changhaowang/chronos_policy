"""Conveyor: a UR5e with a WSG50 finray gripper picks a cube off a belt into a bin.

The belt is static geometry. Once per 20 Hz control step the cube is
teleported forward by ``speed / 20`` and given the belt velocity, after which
it slides under friction, so it effectively travels about 25% faster than the
nominal speed. The arm tracks end-effector targets through mink IK and joint
position servos; physics runs at 500 Hz and observations are recorded at
100 Hz. Each scene seed samples the belt speed, a small belt/bin layout
offset and the cube's spawn pose.

This module reproduces the paper's MuJoCo evaluation (bit for bit with
mink 0.0.12 and NumPy 2.2.6), including two artifacts of the original stack
that change outcomes:

* stepping is dm_control's legacy ``mj_step2`` + ``mj_step1``, and state
  written between steps (resets, belt teleports) is never followed by a
  forward pass, so the first step after a write sees stale derived quantities.
  Episodes therefore depend on what ran before them in the same env,
  including the default warm-up reset done at construction.
* the episode clock accumulates 0.002 s per physics step, so the 30 s limit
  is only exceeded after 601 control steps, and the 0.5 s post-success hold
  is measured on the same drifting clock.
"""

from pathlib import Path

import cv2
from dm_control import mjcf
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums
import mink
import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"
CONTROL_HZ = 20
SOURCE_HZ = 100  # dense observation clock
CAMERA = "third_person_camera"
RENDER_SIZE = 256  # the camera renders at 256 px ...
IMAGE_SIZE = 224  # ... and the dataset stores area-resized 224 px frames

# Scene layout (world frame; the belt moves cubes along +y).
CONVEYOR_CENTER = (0.20, 0.24, 0.22)
CONVEYOR_HALF_EXTENTS = (0.18, 0.60, 0.01)
SPAWN_X = (0.12, 0.28)
SPAWN_Y = (-0.304, -0.304)  # the start edge of the belt
BORDER_CLEARANCE = 0.02
SPAWN_HEIGHT_CLEARANCE = 0.002
BIN_CENTER = (0.50, 0.50, 0.24)
BIN_HALF_EXTENTS = (0.09, 0.07, 0.06)
BIN_HEIGHT_TOLERANCE = 0.02
CUBE_RADIUS = 0.0354
CUBE_CENTER_HEIGHT = 0.025
PICKED_HEIGHT = 0.06  # above the belt the belt stops driving the cube
RESPAWN_Y_MARGIN = 0.03

# Per-episode randomization, as in the data collection.
SPEED_RANGE = (0.01, 0.30)
LAYOUT_OFFSET_X = (-0.02, 0.02)
LAYOUT_OFFSET_Y = (-0.01, 0.01)
YAW_RANGE = (-0.25, 0.25)
INITIAL_TCP = (0.309, 0.24, 0.5317, 0.1382, -0.6863, -0.712, 0.055)
OPEN_GRIPPER = 0.10

# Construction-time defaults; the warm-up reset uses these.
DEFAULT_SPEED = 0.18
DEFAULT_CUBE_POSE = (0.12, 0.06, 0.257, 1.0, 0.0, 0.0, 0.0)
DEFAULT_TCP = (0.34, 0.24, 0.50, 0.225, -0.676, -0.6708, 0.206)

# Robot.
ROBOT_BASE = (0.82, 0.24, 0.2, 0.0, 0.0, 0.0, 1.0)  # xyz + wxyz
HOME_QPOS = (-1.5708, -1.5708, 1.5708, 3.14159, -1.5708, 0.0)
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
GRIPPER_JOINTS = ("right_driver_joint", "left_driver_joint")
IK_DT = 0.002
IK_ITERATIONS = 20
IK_VELOCITY_TOLERANCE = 1e-4
IK_DAMPING = 1e-3
IK_TARGET = "ur5e_wsg50_target"
STABLE_VELOCITY = 2e-5
STABLE_MAX_CHECKS = 1000

# Success judge.
LIFT_HEIGHT = 0.12
PLACE_HEIGHT = 0.04
GRASP_DISTANCE = 0.08
FALL_MARGIN = 0.01
SUCCESS_HOLD_S = 0.5
TIME_LIMIT_S = 30.0


# ------------------------------------------------------------------ math
def qmult(a, b):
    return np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )


def relative_pose(pose, frame):
    """``pose`` (xyz + wxyz) expressed in ``frame``; both in the same frame."""
    inverse = np.array([frame[3], -frame[4], -frame[5], -frame[6]])
    offset = np.zeros(4, pose.dtype)
    offset[1:] = pose[:3] - frame[:3]
    result = np.zeros(7, pose.dtype)
    result[:3] = qmult(qmult(inverse, offset), frame[3:])[1:]
    quat = qmult(inverse, pose[3:])
    result[3:] = -quat if quat[0] < 0 else quat
    return result


def quat_to_rot6d(q):
    """wxyz quaternion -> the first two rows of its rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * w * z,
            2 * x * z + 2 * w * y,
            2 * x * y + 2 * w * z,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * w * x,
        ]
    )


def rot6d_to_quat(rot6d):
    """First two rotation-matrix rows -> positive-w wxyz quaternion (Gram-Schmidt)."""
    a1, a2 = rot6d[:3], rot6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    m = np.zeros((3, 3))
    m[0], m[1], m[2] = b1, b2, np.cross(b1, b2)
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = (0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    return -q if q[0] < 0 else q


def pose10_to_action(pose10):
    """xyz + rot6d + gripper width -> (xyz + wxyz target, gripper width)."""
    pose10 = np.asarray(pose10, dtype=np.float64)
    return np.concatenate([pose10[:3], rot6d_to_quat(pose10[3:9])]), pose10[9:10].copy()


def proprio10(tcp, gripper):
    """robot0_10d: tcp xyz, two rotation rows, gripper width (m)."""
    return np.concatenate([tcp[:3], quat_to_rot6d(tcp[3:]), gripper[:1]])


# --------------------------------------------------------------- layout
class Layout:
    """Belt and bin placement after a per-episode xy offset."""

    def __init__(self, offset=(0.0, 0.0)):
        offset = np.array(offset, dtype=np.float64)
        self.conveyor = np.array(CONVEYOR_CENTER)
        self.conveyor[:2] += offset[:2]
        self.bin = np.array(BIN_CENTER)
        self.bin[:2] += offset[:2]  # the bin follows the belt
        self.half = np.array(CONVEYOR_HALF_EXTENTS)
        self.bin_half = np.array(BIN_HALF_EXTENTS)
        self.spawn_x = np.array(SPAWN_X) + offset[0]
        self.spawn_y = np.array(SPAWN_Y) + offset[1]

    @property
    def end_y(self):
        return float(self.conveyor[1] + self.half[1])

    @property
    def top_z(self):
        return float(self.conveyor[2] + self.half[2])

    def on_belt(self, xyz):
        return bool(
            np.abs(xyz[0] - self.conveyor[0]) <= self.half[0]
            and np.abs(xyz[1] - self.conveyor[1]) <= self.half[1]
        )

    def in_bin_xy(self, xyz):
        return bool(
            np.abs(xyz[0] - self.bin[0]) <= self.bin_half[0]
            and np.abs(xyz[1] - self.bin[1]) <= self.bin_half[1]
        )

    def in_bin(self, xyz):
        return self.in_bin_xy(xyz) and bool(
            np.abs(xyz[2] - self.bin[2]) <= self.bin_half[2] + BIN_HEIGHT_TOLERANCE
        )

    def sample_cube(self, rng):
        """Spawn at the belt's start edge: draws x, y, then yaw."""
        margin = CUBE_RADIUS + BORDER_CLEARANCE
        x_low = max(float(self.spawn_x[0]), self.conveyor[0] - self.half[0] + margin)
        x_high = min(float(self.spawn_x[1]), self.conveyor[0] + self.half[0] - margin)
        y_low = max(float(self.spawn_y[0]), self.conveyor[1] - self.half[1] + margin)
        y_high = min(float(self.spawn_y[1]), self.conveyor[1] + self.half[1] - margin)
        x = rng.uniform(x_low, x_high)
        y = rng.uniform(y_low, y_high)
        yaw = rng.uniform(*YAW_RANGE)
        z = self.top_z + CUBE_CENTER_HEIGHT + SPAWN_HEIGHT_CLEARANCE
        return np.array([x, y, z, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def sample_episode(seed, speed_range=SPEED_RANGE):
    """Scene of one seed, with the collection's draw order."""
    rng = np.random.default_rng(seed)
    speed = rng.uniform(float(min(speed_range)), float(max(speed_range)))
    offset = np.array([rng.uniform(*LAYOUT_OFFSET_X), rng.uniform(*LAYOUT_OFFSET_Y)])
    cube = Layout(offset).sample_cube(rng)
    tcp = np.array(INITIAL_TCP)
    rng.random(1)  # convex weights over the single start-pose candidate
    tcp[3:] = tcp[3:] / np.linalg.norm(tcp[3:])
    rng.random(1)  # ... and over the single gripper-width candidate
    return dict(seed=int(seed), speed=float(speed), offset=offset, cube=cube, tcp=tcp)


# ------------------------------------------------------------------ model
def load_robot():
    """UR5e with the WSG50 finray gripper on its flange."""
    arm = mjcf.from_path(str(ASSET_DIR / "ur5e" / "ur5e.xml"))
    arm.model = "ur5e"
    del arm.keyframe
    arm.find("site", "attachment_site").attach(
        mjcf.from_path(str(ASSET_DIR / "wsg50" / "wsg50_finray_milibar.xml"))
    )
    return arm


def build_physics():
    world = mjcf.from_path(str(ASSET_DIR / "scene" / "table_conveyor_pick_place.xml"))
    site = world.worldbody.add(
        "site", name="ur5e_site", pos=np.array(ROBOT_BASE[:3]), quat=np.array(ROBOT_BASE[3:])
    )
    site.attach(load_robot())
    cube = mjcf.from_path(str(ASSET_DIR / "cube" / "conveyor_cube.xml"))
    cube.model = "cube"
    site = world.worldbody.add(
        "site",
        name="cube_site",
        pos=np.array(DEFAULT_CUBE_POSE[:3]),
        quat=np.array(DEFAULT_CUBE_POSE[3:]),
        group=3,
    )
    site.attach(cube).add("freejoint")
    return mjcf.Physics.from_mjcf_model(world)


class ArmIK:
    """mink IK on a separate robot-only model, expressed in the robot base frame."""

    def __init__(self):
        robot = load_robot()
        target = robot.worldbody.add("body", name=IK_TARGET, mocap=True)
        target.add(
            "geom",
            type="box",
            size=[0.005, 0.005, 0.005],
            contype=0,
            conaffinity=0,
            rgba=[1, 1, 1, 0.2],
        )
        self.physics = mjcf.Physics.from_mjcf_model(robot)
        model, self.data = self.physics.model.ptr, self.physics.data.ptr
        self.joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.actuators = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS
        ]
        self.configuration = mink.Configuration(model=model)
        self.task = mink.FrameTask(
            frame_name="wsg50/end_effector",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.task.set_target(mink.SE3.from_mocap_name(model, self.data, IK_TARGET))
        self.mocap = model.body(IK_TARGET).mocapid[0]
        self.model = model

    def solve(self, tcp, qpos, ctrl, qvel):
        """Joint targets for a world-frame tcp pose, warm-started at ``qpos``.

        No velocity limit is passed, so mink applies only joint-range limits.
        """
        local = relative_pose(tcp, np.array(ROBOT_BASE))
        data = self.data
        data.qpos[self.joints] = qpos
        data.ctrl[self.actuators] = ctrl
        data.qvel[self.joints] = qvel
        data.qacc[self.joints] = 0.0
        data.qfrc_applied[self.joints] = 0.0
        data.mocap_pos[self.mocap] = local[:3]
        data.mocap_quat[self.mocap] = local[3:]
        self.task.set_target(mink.SE3.from_mocap_name(self.model, data, IK_TARGET))
        self.configuration.update(data.qpos)
        for _ in range(IK_ITERATIONS):
            velocity = mink.solve_ik(
                self.configuration, [self.task], IK_DT, solver="quadprog", damping=IK_DAMPING
            )
            self.configuration.integrate_inplace(velocity, IK_DT)
            if np.linalg.norm(velocity) < IK_VELOCITY_TOLERANCE:
                break
        return self.configuration.q[self.joints].copy()


class Judge:
    """Pick, place and failure detection on end-of-step observations."""

    def reset(self):
        self.picked = self.placed = self.missed = self.fell = self.timed_out = False
        self.placed_at = None
        self.since_placed = None

    def update(self, layout, cube, tcp, time):
        if self.placed_at is not None:
            self.since_placed = time - self.placed_at
        if time > TIME_LIMIT_S:
            self.timed_out = True
        if (
            not layout.on_belt(cube[:3])
            and not layout.in_bin(cube[:3])
            and cube[2] <= layout.top_z - FALL_MARGIN
        ):
            self.fell = True
            return
        if not self.picked:
            if cube[1] >= layout.end_y:
                self.missed = True
                return
            if (
                cube[2] > layout.top_z + LIFT_HEIGHT
                and np.linalg.norm(cube[:3] - tcp[:3]) < GRASP_DISTANCE
            ):
                self.picked = True
            return
        if not self.placed and layout.in_bin_xy(cube) and cube[2] < layout.bin[2] + PLACE_HEIGHT:
            self.placed = True
            self.placed_at = time

    @property
    def reward(self):
        return 0.0 if self.timed_out else float(self.picked and self.placed)

    @property
    def done(self):
        if self.timed_out or self.missed or self.fell:
            return True
        return self.since_placed is not None and self.since_placed > SUCCESS_HOLD_S

    def states(self):
        return dict(picked=self.picked, placed=self.placed, missed=self.missed, fell=self.fell)


class ConveyorEnv:
    """The paper's conveyor scene; ``reset`` then ``step`` 20 Hz targets."""

    def __init__(self, render=True):
        self.render_images = render
        self.physics = build_physics()
        model = self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.timestep = model.opt.timestep
        self.substeps = int(1 / CONTROL_HZ / self.timestep)  # 25
        self.substep_dt = 1.0 / CONTROL_HZ / self.substeps
        self.record_every = self.substeps // (SOURCE_HZ // CONTROL_HZ)  # 5
        self.arm_qpos = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ur5e/{n}") for n in ARM_JOINTS
        ]
        self.arm_ctrl = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ur5e/{n}")
            for n in ARM_ACTUATORS
        ]
        self.gripper_qpos = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"ur5e/wsg50/{n}")
            for n in GRIPPER_JOINTS
        ]
        self.gripper_ctrl = [model.actuator("ur5e/wsg50/gripper").id]
        self.tcp_site = model.site("ur5e/wsg50/end_effector").id
        self.camera = model.camera(CAMERA).id
        cube = model.joint("cube/")
        self.cube_qpos, self.cube_dof = int(cube.qposadr[0]), int(cube.dofadr[0])
        self.conveyor_body = model.body("conveyor_frame").id
        self.bin_body = model.body("target_bin").id
        self.conveyor_body_pos = model.body_pos[self.conveyor_body].copy()
        self.bin_body_pos = model.body_pos[self.bin_body].copy()
        self.scene_option = wrapper.MjvOption()
        self.scene_option.frame = enums.mjtFrame.mjFRAME_NONE
        self.ik = ArmIK()
        self.judge = Judge()
        self.gripper_width = OPEN_GRIPPER
        self.speed = DEFAULT_SPEED
        self.rng = np.random.default_rng(0)
        # The original builds the simulator with a default reset; its state
        # carries into the first episode.
        self._reset(
            Layout(), np.array(DEFAULT_CUBE_POSE), np.array(DEFAULT_TCP), np.array([OPEN_GRIPPER])
        )

    # ------------------------------------------------------------ episodes
    def reset(self, episode):
        """Reset to a ``sample_episode`` scene; returns the t = 0 observation."""
        self.rng = np.random.default_rng(episode["seed"])  # cube respawns
        self.speed = episode["speed"]
        self._reset(
            Layout(episode["offset"]),
            np.array(episode["cube"]),
            np.array(episode["tcp"]),
            np.array([OPEN_GRIPPER]),
        )
        frame = self.observe(0.0)
        self.judge.update(self.layout, self.cube_pose, self.tcp_pose, self.time)
        return frame

    def _reset(self, layout, cube, tcp, gripper):
        self.layout = layout
        model = self.model
        model.body_pos[self.conveyor_body] = self.conveyor_body_pos + (
            layout.conveyor - np.array(CONVEYOR_CENTER)
        )
        model.body_pos[self.bin_body] = self.bin_body_pos + (layout.bin - np.array(BIN_CENTER))
        self._set_cube(cube)
        self._set_joints(np.array(HOME_QPOS), np.array([self.gripper_width]))
        self._set_joints(self._solve_ik(tcp), gripper)
        self.last_tcp = tcp.copy()
        self._wait_until_stable()
        self.time = 0.0
        self.judge.reset()
        if self.layout.on_belt(self.cube_pose[:3]) and not self._picked(self.cube_pose):
            self._set_cube_velocity()

    def _set_joints(self, qpos, gripper):
        data = self.data
        data.qpos[self.arm_qpos] = qpos
        data.ctrl[self.arm_ctrl] = qpos
        data.qvel[self.arm_qpos] = 0.0
        data.qacc[self.arm_qpos] = 0.0
        data.qfrc_applied[self.arm_qpos] = 0.0
        self.gripper_width = gripper[0]
        data.qpos[self.gripper_qpos] = gripper / 2
        data.ctrl[self.gripper_ctrl] = gripper / 2
        data.qvel[self.gripper_qpos] = 0.0
        data.qacc[self.gripper_qpos] = 0.0
        data.qfrc_applied[self.gripper_qpos] = 0.0

    def _wait_until_stable(self):
        previous = self.data.qpos.copy()
        for _ in range(STABLE_MAX_CHECKS + 1):
            for _ in range(10):
                self.physics.step()
            current = self.data.qpos.copy()
            if np.linalg.norm((current - previous) / len(current)) < STABLE_VELOCITY:
                return
            previous = current
        raise RuntimeError("The scene did not settle")

    def _solve_ik(self, tcp):
        data = self.data
        return self.ik.solve(
            tcp, data.qpos[self.arm_qpos], data.ctrl[self.arm_ctrl], data.qvel[self.arm_qpos]
        )

    # -------------------------------------------------------------- state
    @property
    def tcp_pose(self):
        pose = np.zeros(7)
        pose[:3] = self.data.site_xpos[self.tcp_site]
        mujoco.mju_mat2Quat(pose[3:], self.data.site_xmat[self.tcp_site])
        pose[3:] = pose[3:] / np.linalg.norm(pose[3:])
        if pose[3] < 0:
            pose[3:] = -pose[3:]
        return pose

    @property
    def gripper_opening(self):
        return np.array(
            [self.data.qpos[self.gripper_qpos[0]] + self.data.qpos[self.gripper_qpos[1]]]
        )

    @property
    def cube_pose(self):
        return self.data.qpos[self.cube_qpos : self.cube_qpos + 7].copy()

    def _set_cube(self, pose):
        self.data.qpos[self.cube_qpos : self.cube_qpos + 7] = pose
        self.data.qvel[self.cube_dof : self.cube_dof + 6] = 0
        self.data.qacc[self.cube_dof : self.cube_dof + 6] = 0

    def _set_cube_velocity(self):
        self.data.qvel[self.cube_dof : self.cube_dof + 6] = np.array(
            [0.0, self.speed, 0.0, 0.0, 0.0, 0.0]
        )

    def _picked(self, cube):
        return bool(cube[2] >= self.layout.top_z + PICKED_HEIGHT)

    def render(self):
        image = self.physics.render(
            width=RENDER_SIZE,
            height=RENDER_SIZE,
            camera_id=self.camera,
            scene_option=self.scene_option,
        )
        return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    def observe(self, time):
        """One dense-source sample: time, robot0_10d (float64) and the 224 px image."""
        return dict(
            time=time,
            proprio=proprio10(self.tcp_pose, self.gripper_opening),
            image=self.render() if self.render_images else None,
        )

    @property
    def executed_action(self):
        return self.last_tcp.copy(), np.array([self.gripper_width])

    # ------------------------------------------------------------ control
    def step(self, tcp, gripper):
        """Track a 7-D tcp target for one 20 Hz step; returns 5 samples at 100 Hz.

        Joint servo targets ramp linearly to the IK solution, reaching 24/25
        of it by the step's last physics tick; the gripper target is immediate.
        """
        tcp = np.asarray(tcp, dtype=np.float64)
        target = self._solve_ik(tcp)
        data = self.data
        data.ctrl[self.gripper_ctrl] = gripper[0] / 2
        self.gripper_width = gripper[0]
        start = data.ctrl[self.arm_ctrl]
        self.last_tcp = tcp.copy()
        frames = []
        for i in range(self.substeps):
            data.ctrl[self.arm_ctrl] = start + (target - start) * i / self.substeps
            self.physics.step()
            self.time += self.substep_dt
            if (i + 1) % self.record_every == 0:
                frames.append(self.observe(self.time))
        cube, layout = self.cube_pose, self.layout
        self.judge.update(layout, cube, self.tcp_pose, self.time)
        # The belt acts after the step is observed.
        if layout.on_belt(cube[:3]) and not self._picked(cube):
            margin = CUBE_RADIUS + BORDER_CLEARANCE
            cube[0] = np.clip(
                cube[0],
                layout.conveyor[0] - layout.half[0] + margin,
                layout.conveyor[0] + layout.half[0] - margin,
            )
            cube[1] += self.speed / CONTROL_HZ
            self._set_cube(cube)
            self._set_cube_velocity()
            cube = self.cube_pose
        if self._should_respawn(cube):
            self._set_cube(layout.sample_cube(self.rng))
            self._set_cube_velocity()
        return frames

    def _should_respawn(self, cube):
        """Respawn a cube dropped off the belt; one at the belt's end is a miss instead."""
        layout = self.layout
        if self._picked(cube) or layout.in_bin(cube[:3]) or cube[1] >= layout.end_y:
            return False
        return cube[1] > layout.end_y + RESPAWN_Y_MARGIN or not layout.on_belt(cube[:3])

    @property
    def done(self):
        return self.judge.done

    @property
    def success(self):
        return self.judge.reward >= 1.0

    def close(self):
        self.physics.free()
        self.ik.physics.free()
