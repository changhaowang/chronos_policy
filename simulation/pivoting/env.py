"""Pivoting: a floating WSG50 gripper pivots a book upright against a bookend.

The gripper is a free rigid body driven by a Cartesian impedance wrench, with
gravity cancelled per body. Physics runs at 1 kHz. Each episode samples the
book's size, mass, colour and the gripper's start pose; the bookend placement
is fixed by the scene seed.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import cv2
from dm_control import mjcf
from dm_control.mujoco import wrapper
from dm_control.mujoco.engine import MovableCamera
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSET_DIR = Path(__file__).resolve().parent / "assets"
TIMESTEP = 0.001
IMAGE_SIZE = 224

TOOL_KP = 4000.0  # translational impedance used for the nominal (non-adaptive) controller
TOOL_ROT_KP = 300.0
DAMPING_RATIO = 1.05
PERPENDICULAR_STIFFNESS = 4000.0
STIFFNESS_RANGE = (2000.0, 16000.0)
SURFACE_FORCE_LIMIT = 80.0  # caps spring energy stored against the table/bookend
BOOK_FORCE_LIMIT = 18.0  # caps spring energy stored against the book
CONTACT_GRACE_STEPS = 20  # 20 ms before a force-limit latch releases
FORCE_SENSOR_CUTOFF_HZ = 30.0
COLLISION_ENVELOPE = (0.18, 0.12, 0.025)  # largest sampled book, for broad-phase bounds
SUCCESS_ANGLE_DEG = 15.0

BOOKEND_X = (0.4, 0.2)  # minimum and span
STANDOFF = 0.05
START_PRISM = (0.04, 0.06, 0.05)  # depth, lateral, vertical
BOOK_COLORS = np.array(
    [
        [0.64, 0.16, 0.14, 1.0],
        [0.13, 0.25, 0.48, 1.0],
        [0.16, 0.43, 0.27, 1.0],
        [0.67, 0.43, 0.12, 1.0],
        [0.12, 0.43, 0.46, 1.0],
        [0.42, 0.25, 0.16, 1.0],
        [0.43, 0.20, 0.43, 1.0],
        [0.32, 0.36, 0.40, 1.0],
    ]
)
TIP_PADS = ("wsg50/right_tip_pad", "wsg50/left_tip_pad")
GUARDED_SURFACES = (
    "table/table_surface",
    "bookend2_blender/robot_wall_surface",
    "bookend2_blender/robot_pivot_surface",
    "bookend2_blender/robot_floor_surface",
)


@dataclass(frozen=True)
class Book:
    mass_kg: float = 1.375
    sliding_friction: float = 0.12
    torsional_friction: float = 0.002
    rolling_friction: float = 0.0001
    length_m: float = 0.15
    width_m: float = 0.10
    thickness_m: float = 0.025

    @property
    def friction(self):
        return (self.sliding_friction, self.torsional_friction, self.rolling_friction)


NOMINAL_BOOK = Book()


def rpy(roll, pitch, yaw):
    """Rotation matrix Rz(yaw) Ry(pitch) Rx(roll) from angles in degrees."""
    roll, pitch, yaw = np.deg2rad(np.array([roll, pitch, yaw], dtype=float))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def apply(transform_matrix, point):
    homogeneous = transform_matrix @ np.r_[point, 1.0].reshape(4, 1)
    return (homogeneous[:3] / homogeneous[3]).reshape(3)


def wxyz(rotation_matrix):
    return Rotation.from_matrix(rotation_matrix).as_quat()[[3, 0, 1, 2]]


def tool_orientation(position):
    """Tool pitched 30 degrees down and yawed away from the (virtual) robot base."""
    delta = np.asarray(position, dtype=float)[:2] - np.array([-0.3, 0.0])
    if abs(delta[0]) <= 1e-5:
        raise ValueError("Tool target is too close to the robot base x-coordinate")
    return rpy(0.0, -30.0, np.degrees(np.arctan(delta[1] / delta[0])))


def pose9_to_pose7(pose):
    """xyz + first two rotation-matrix rows -> xyz + wxyz quaternion."""
    pose = np.asarray(pose, dtype=np.float64)
    a = pose[3:6] / max(np.linalg.norm(pose[3:6]), 1e-12)
    b = pose[6:9] - a * np.dot(a, pose[6:9])
    b = b / max(np.linalg.norm(b), 1e-12)
    matrix = np.stack([a, b, np.cross(a, b)])
    if abs(np.linalg.det(matrix) - 1) > 1e-5:
        raise ValueError("Degenerate predicted rotation")
    return np.r_[pose[:3], Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]]


def pose7_to_pose9(pose):
    pose = np.asarray(pose)
    matrix = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return np.r_[pose[:3], matrix[:2].reshape(-1)].astype(np.float32)


def layout(book, seed=0):
    """Bookend and book placement plus the tool positions the flip pivots about."""
    rng = np.random.RandomState(seed)
    bookend = transform(
        rpy(90.0, 0.0, 180.0 + rng.uniform(-10.0, 10.0)),
        [BOOKEND_X[0] + rng.uniform(0.0, BOOKEND_X[1]), rng.uniform(-0.2, 0.2), 0.2],
    )
    book_relative = transform(rpy(90.0, 0.0, 0.0), [0.015, 0.035, 0.03])
    length = book.length_m + 0.01
    upper = book.thickness_m * 0.30
    lower = book.thickness_m * 0.70
    unit_x, unit_y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    tip = apply(book_relative, np.array([book.length_m, book.width_m / 2.0, upper]))
    engage = np.array([book.thickness_m * 0.28 + length, tip[1], tip[2]])
    angle = 20 / 20.0 * np.pi / 2.0
    upright = (
        engage
        + (upper * np.sin(angle) + length * np.cos(angle) - length) * unit_x
        + (lower * np.cos(angle) + length * np.sin(angle) - lower) * unit_y
    )
    return {
        "bookend_transform": bookend,
        "book_transform": bookend @ book_relative,
        "bookend_rotation": bookend[:3, :3],
        "prepare": apply(bookend, tip + STANDOFF * unit_x),
        "engage": apply(bookend, engage),
        "upright": apply(bookend, upright),
    }


def sample_book(rng, jitter=0.2):
    """Vary the book's height, width and mass by up to 20 percent."""
    for _ in range(100):
        length = NOMINAL_BOOK.length_m * rng.uniform(1.0 - jitter, 1.0 + jitter)
        width = NOMINAL_BOOK.width_m * rng.uniform(1.0 - jitter, 1.0 + jitter)
        if length > width + 1e-4:
            break
    mass = NOMINAL_BOOK.mass_kg * rng.uniform(1.0 - jitter, 1.0 + jitter)
    return Book(mass_kg=float(mass), length_m=float(length), width_m=float(width))


def sample_color(rng):
    color = BOOK_COLORS[int(rng.integers(len(BOOK_COLORS)))].copy()
    color[:3] = np.clip(color[:3] * rng.uniform(0.88, 1.12), 0.08, 0.82)
    return color


def sample_start(scene, rng):
    """Start in a prism in front of the book: 70% near its centre, 30% uniform."""
    if rng.random() < 0.7:
        normalized = np.clip(rng.normal(0.0, 0.22, size=3), -0.5, 0.5)
    else:
        normalized = rng.uniform(-0.5, 0.5, size=3)
    depth, lateral, vertical = normalized * np.asarray(START_PRISM)
    return scene["prepare"] + scene["bookend_rotation"] @ np.array([depth, vertical, lateral])


def compliance_matrix(nominal, virtual, stiffness):
    """Adaptive stiffness along the nominal-to-virtual direction, fixed elsewhere."""
    direction = np.asarray(virtual[:3]) - np.asarray(nominal[:3])
    length = np.linalg.norm(direction)
    if length < 0.001:
        return PERPENDICULAR_STIFFNESS * np.eye(3)
    direction /= length
    return PERPENDICULAR_STIFFNESS * np.eye(3) + (
        float(stiffness) - PERPENDICULAR_STIFFNESS
    ) * np.outer(direction, direction)


def _attach(world, child, transform_matrix, *, freejoint):
    site = world.worldbody.add(
        "site",
        name=f"{child.model}_attachment_site",
        pos=transform_matrix[:3, 3],
        quat=wxyz(transform_matrix[:3, :3]),
        group=3,
    )
    body = site.attach(child)
    if freejoint:
        body.add("freejoint")


def _physical_geoms(model):
    return tuple(
        geom
        for geom in model.find_all("geom")
        if getattr(geom.dclass, "full_identifier", "") != "visual"
    )


def build_model(scene):
    """Assemble the scene. Contact bits: book 1, visible support 2, table 4."""
    world = mjcf.from_path(str(ASSET_DIR / "ground.xml"))
    gripper = mjcf.from_path(str(ASSET_DIR / "wsg50" / "wsg50.xml"))
    gripper_geoms = _physical_geoms(gripper)
    camera = mjcf.from_path(str(ASSET_DIR / "realsense_d435i" / "d435i_with_cam.xml"))
    robot_geoms = gripper_geoms + _physical_geoms(camera)
    gripper.find("site", "cam_mount").attach(camera)
    # The free-joint pose is overwritten after compilation; this start keeps
    # the compiler's inferred inertias and bounds well behaved.
    _attach(world, gripper, transform(np.eye(3), scene["prepare"]), freejoint=True)

    table = mjcf.from_path(str(ASSET_DIR / "table" / "table.xml"))
    table_surface = table.find("geom", "table_surface")
    world.worldbody.add("site", name="table_attachment_site", pos=(0.5, 0.0, 0.0)).attach(table)

    bookend = mjcf.from_path(str(ASSET_DIR / "bookend" / "bookend2_blender.xml"))
    surface_names = {"robot_wall_surface", "robot_pivot_surface", "robot_floor_surface"}
    bookend_geoms = _physical_geoms(bookend)
    robot_surfaces = tuple(g for g in bookend_geoms if g.name in surface_names)
    supports = tuple(g for g in bookend_geoms if g.name not in surface_names)
    _attach(world, bookend, scene["bookend_transform"], freejoint=False)

    book = mjcf.from_path(str(ASSET_DIR / "book" / "book2_blend.xml"))
    book_collision = book.find("geom", "book_collision")
    half = np.asarray(COLLISION_ENVELOPE) / 2.0
    book_collision.pos = tuple(half)
    book_collision.size = tuple(half)
    book_collision.mass = NOMINAL_BOOK.mass_kg
    book_collision.friction = NOMINAL_BOOK.friction
    _attach(world, book, scene["book_transform"], freejoint=True)

    frame = mjcf.from_path(str(ASSET_DIR / "frame" / "frame.xml"))
    _attach(world, frame, transform(np.eye(3), (0.5, 1.0, 0.5)), freejoint=False)

    for geom in world.find_all("geom"):
        geom.contype = 0
        geom.conaffinity = 0
    book_collision.contype = 1
    for geom in supports:
        geom.conaffinity = 1
    for geom in robot_geoms:
        geom.conaffinity = 5
    for geom in gripper_geoms:
        geom.conaffinity = 7
    for geom in robot_surfaces:
        geom.contype = 2
    table_surface.contype = 4
    return mjcf.Physics.from_mjcf_model(world)


class PivotingEnv:
    """Floating-gripper pivoting scene with collection-matched resets."""

    def __init__(self, scene_seed=0):
        self.scene_seed = int(scene_seed)
        self.scene = layout(NOMINAL_BOOK, self.scene_seed)
        self.physics = build_model(self.scene)
        self.model, self.data = self.physics.model, self.physics.data
        if not np.isclose(self.model.opt.timestep, TIMESTEP, rtol=0.0, atol=1e-12):
            raise ValueError(f"Pivoting requires a {TIMESTEP} s physics timestep")
        tau = 1.0 / (2.0 * np.pi * FORCE_SENSOR_CUTOFF_HZ)
        self._filter_alpha = float(1.0 - np.exp(-TIMESTEP / tau))

        model = self.model
        self._gripper_qpos = np.array(
            [model.joint(f"wsg50/{side}_driver_joint").qposadr.item() for side in ("right", "left")]
        )
        self._gripper_dofs = np.array(
            [model.joint(f"wsg50/{side}_driver_joint").dofadr.item() for side in ("right", "left")]
        )
        self._gripper_actuator = model.actuator("wsg50/gripper").id
        self._tool_site = model.site("wsg50/planner_tip_site").id
        self._book_body = model.body("book2_blend/book2_blend").id
        self._book_geom = model.geom("book2_blend/book_collision").id
        self._tip_geoms = frozenset(model.geom(name).id for name in TIP_PADS)
        # Softer fingertip pads: 20 ms time constant, damping ratio 2, 5 mm width.
        for geom in self._tip_geoms:
            model.geom_solref[geom] = (0.020, 2.0)
            model.geom_solimp[geom, 2] = 0.005
        free_joint = model.joint("wsg50/").id
        self._free_qpos = int(model.jnt_qposadr[free_joint])
        free_body = int(model.jnt_bodyid[free_joint])
        self._gripper_base = model.body("wsg50/base").id
        root = int(model.body_rootid[free_body])
        self._robot_bodies = frozenset(
            i for i in range(model.nbody) if int(model.body_rootid[i]) == root
        )
        self._guarded = frozenset(model.geom(name).id for name in GUARDED_SURFACES)
        self._contact = np.zeros(6)

        # Critically damped gains from the gripper's mass and inertia.
        mass = sum(float(model.body_mass[i]) for i in self._robot_bodies)
        com = sum(
            float(model.body_mass[i]) * np.asarray(self.data.xipos[i]) for i in self._robot_bodies
        ) / max(mass, 1e-9)
        inertia = np.zeros((3, 3))
        for body in self._robot_bodies:
            body_mass = float(model.body_mass[body])
            if body_mass <= 0.0:
                continue
            rotation = np.asarray(self.data.ximat[body]).reshape(3, 3)
            inertia += rotation @ np.diag(model.body_inertia[body]) @ rotation.T
            offset = np.asarray(self.data.xipos[body]) - com
            inertia += body_mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
        principal = np.linalg.eigvalsh(inertia)
        self._rot_kd = float(
            2.0 * DAMPING_RATIO * np.sqrt(TOOL_ROT_KP * max(float(np.max(principal)), 1e-8))
        )
        self._kd = 2.0 * DAMPING_RATIO * np.sqrt(TOOL_KP * max(mass, 1e-6))

        model.vis.global_.offwidth = max(640, int(model.vis.global_.offwidth))
        model.vis.global_.offheight = max(480, int(model.vis.global_.offheight))

        # Per-episode book resizing edits the compiled mesh and collision box.
        self._book_visual = model.geom("book2_blend/book_visual").id
        self._book_mesh = int(model.geom_dataid[self._book_visual])
        start = int(model.mesh_vertadr[self._book_mesh])
        self._mesh_slice = slice(start, start + int(model.mesh_vertnum[self._book_mesh]))
        self._mesh_reference = np.asarray(model.mesh_vert[self._mesh_slice], dtype=float).copy()
        self._mesh_reference_size = np.array(
            [NOMINAL_BOOK.thickness_m, NOMINAL_BOOK.width_m, NOMINAL_BOOK.length_m]
        )
        self._mesh_version = 0
        self._book_qpos = int(
            model.jnt_qposadr[model.body_jntadr[model.body_parentid[self._book_body]]]
        )
        self._envelope = 2.0 * np.asarray(model.geom_size[self._book_geom], dtype=float)
        self._book_rbound = float(model.geom_rbound[self._book_geom])
        self._book_aabb = np.asarray(model.geom_aabb[self._book_geom], dtype=float).copy()
        self._bvh_aabb = np.asarray(model.bvh_aabb, dtype=float).copy()

        self._initial_qpos = self.data.qpos.copy()
        self.physics.forward()
        self._free_to_tool = np.linalg.inv(
            self._pose(self.data.xpos[free_body], self.data.xmat[free_body])
        ) @ self._pose(self.data.site_xpos[self._tool_site], self.data.site_xmat[self._tool_site])
        self._set_tool_start(self.scene["prepare"])
        self._reset_state()

        # The camera is fixed per scene seed, matching the data collection.
        camera_rng = np.random.default_rng(np.random.SeedSequence([self.scene_seed, 0, 0xF11F]))
        camera_scene = layout(sample_book(camera_rng), self.scene_seed)
        self.camera_lookat = 0.5 * (camera_scene["engage"] + camera_scene["upright"])
        self._camera = None

    @staticmethod
    def _pose(position, matrix):
        return transform(np.asarray(matrix).reshape(3, 3), position)

    # ------------------------------------------------------------------ episodes
    def reset(self, episode_seed):
        """Sample a book and a contact-free start, as in the data collection."""
        rng = np.random.default_rng(
            np.random.SeedSequence([self.scene_seed, int(episode_seed), 0xF11F])
        )
        book = sample_book(rng)
        color = sample_color(rng)
        scene = layout(book, self.scene_seed)
        for _ in range(32):
            start = sample_start(scene, rng)
            self.configure(book, color, start)
            if self.settle_error <= 0.01 and np.linalg.norm(self._raw_wrench_world[:3]) <= 0.5:
                break
        else:
            raise RuntimeError("Unable to sample a contact-free start")
        return dict(
            episode_seed=int(episode_seed),
            scene_seed=self.scene_seed,
            book=asdict(book),
            book_color_rgba=color.tolist(),
            start_position=start.tolist(),
        )

    def configure(self, book, color, start):
        """Apply a book and gripper start position, then reset the physics."""
        model = self.model
        half = np.array([book.length_m, book.width_m, book.thickness_m]) / 2.0
        if np.any(2.0 * half > self._envelope + 1e-12):
            raise ValueError("Book exceeds the compiled collision envelope")
        model.geom_size[self._book_geom] = half
        model.geom_pos[self._book_geom] = half
        model.geom_friction[self._book_geom] = book.friction
        # Mesh axes are thickness, width, length.
        size = np.array([book.thickness_m, book.width_m, book.length_m])
        model.mesh_vert[self._mesh_slice] = self._mesh_reference * (
            size / self._mesh_reference_size
        )
        self._mesh_version += 1
        model.geom_pos[self._book_visual] = half
        model.geom_matid[self._book_visual] = -1
        model.geom_rgba[self._book_visual] = np.clip(np.asarray(color, dtype=float), 0.0, 1.0)
        length, width, thickness = 2.0 * half
        model.body_mass[self._book_body] = book.mass_kg
        model.body_ipos[self._book_body] = half
        model.body_inertia[self._book_body] = (
            book.mass_kg
            / 12.0
            * np.array(
                [
                    length * length + width * width,
                    length * length + thickness * thickness,
                    width * width + thickness * thickness,
                ]
            )
        )
        book_transform = layout(book, self.scene_seed)["book_transform"]
        self._initial_qpos[self._book_qpos : self._book_qpos + 7] = np.r_[
            book_transform[:3, 3], wxyz(book_transform[:3, :3])
        ]
        self._set_tool_start(np.asarray(start, dtype=float))
        mujoco.mj_setConst(model.ptr, self.data.ptr)
        model.geom_rbound[self._book_geom] = self._book_rbound
        model.geom_aabb[self._book_geom] = self._book_aabb
        model.bvh_aabb[:] = self._bvh_aabb
        self._reset_state()

    def _set_tool_start(self, position):
        self._tool_start = np.asarray(position, dtype=float).copy()
        world_from_free = transform(tool_orientation(position), position) @ np.linalg.inv(
            self._free_to_tool
        )
        self._initial_qpos[self._free_qpos : self._free_qpos + 7] = np.r_[
            world_from_free[:3, 3], wxyz(world_from_free[:3, :3])
        ]

    def _reset_state(self):
        data = self.data
        data.qpos[:] = self._initial_qpos
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        self.physics.forward()
        self._close_gripper()
        self._update_wrench(reset=True)
        self._surface_latch = [None, None, 0]  # normal, boundary, missed steps
        self._book_latch = [None, None, 0]
        self.settle_error = float(np.linalg.norm(self.tool_position - self._tool_start))
        data.time = 0.0

    # --------------------------------------------------------------- observations
    @property
    def tool_position(self):
        return np.array(self.data.site_xpos[self._tool_site], dtype=float)

    @property
    def tool_quaternion(self):
        return wxyz(np.array(self.data.site_xmat[self._tool_site]).reshape(3, 3))

    @property
    def tool_pose7(self):
        return np.r_[self.tool_position, self.tool_quaternion]

    def observation(self):
        """Tool pose (xyz + two rotation rows) and the filtered tool-frame wrench."""
        return pose7_to_pose9(self.tool_pose7), self._sensor_wrench.astype(np.float32)

    def render(self):
        """224x224 RGB: rendered at 640x480, then area-resized as in the dataset."""
        if self._camera is None or self._uploaded != self._mesh_version:
            # A fresh camera per book keeps renders identical to the collection.
            self._camera = MovableCamera(self.physics, height=480, width=640)
            self._camera.set_pose(np.asarray(self.camera_lookat, dtype=float), 0.75, -30.0, -25.0)
            self._options = wrapper.MjvOption()
            contexts = self.physics.contexts
            with contexts.gl.make_current() as context:
                context.call(
                    mujoco.mjr_uploadMesh, self.model.ptr, contexts.mujoco.ptr, self._book_mesh
                )
            self._uploaded = self._mesh_version
        image = self._camera.render(
            scene_option=self._options,
            render_flag_overrides={mujoco.mjtRndFlag.mjRND_REFLECTION: False},
        )
        return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    def book_angle_deg(self):
        """Angle between the book's long axis and vertical."""
        s, x, y, z = self.data.xquat[self._book_body]
        return float(np.degrees(np.arccos(np.clip(abs(2 * (x * z - s * y)), -1.0, 1.0))))

    def success(self):
        return self.book_angle_deg() < SUCCESS_ANGLE_DEG

    # ------------------------------------------------------------------- control
    def step(self, nominal, virtual, stiffness):
        """Advance 1 ms toward a 7-D virtual target with adaptive stiffness.

        ``nominal`` and ``virtual`` are xyz + wxyz poses; stiffness acts along
        the line between them.
        """
        stiffness = float(np.clip(stiffness, *STIFFNESS_RANGE))
        matrix = compliance_matrix(nominal, virtual, stiffness)
        # Force limits use the stiffest direction so adaptation cannot loosen them.
        kp = float(np.linalg.eigvalsh(matrix).max())
        target = np.asarray(virtual[:3]).copy()
        target = self._limit(
            self._surface_latch, self._surface_normal(), target, SURFACE_FORCE_LIMIT / kp
        )
        target = self._limit(self._book_latch, self._book_normal(), target, BOOK_FORCE_LIMIT / kp)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model.ptr, self.data.ptr, mujoco.mjtObj.mjOBJ_SITE, self._tool_site, velocity, 0
        )
        force = matrix @ (target - self.tool_position) - self._kd * velocity[3:]
        # Keep the predicted orientation residual on top of the position-derived
        # orientation used during collection.
        command = Rotation.from_matrix(tool_orientation(nominal[:3]))
        drive = Rotation.from_matrix(tool_orientation(target))
        predicted = Rotation.from_quat(np.asarray(virtual)[[4, 5, 6, 3]])
        current = Rotation.from_quat(self.tool_quaternion[[1, 2, 3, 0]])
        rotation = drive * command.inv() * predicted
        torque = TOOL_ROT_KP * (rotation * current.inv()).as_rotvec() - self._rot_kd * velocity[:3]

        data = self.data
        data.xfrc_applied[:] = 0.0
        gravity = np.asarray(self.model.opt.gravity, dtype=float)
        for body in self._robot_bodies:
            body_mass = float(self.model.body_mass[body])
            if body_mass > 0.0:
                data.xfrc_applied[body, :3] -= body_mass * gravity
        data.xfrc_applied[self._gripper_base, :3] += force
        data.xfrc_applied[self._gripper_base, 3:] += torque
        data.ctrl[self._gripper_actuator] = 0.0
        mujoco.mj_step(self.model.ptr, data.ptr)
        self._close_gripper()
        self._update_wrench()

    def _close_gripper(self):
        """The fingers stay closed: project them back after every step."""
        self.data.qpos[self._gripper_qpos] = 0.0
        self.data.qvel[self._gripper_dofs] = 0.0
        self.data.ctrl[self._gripper_actuator] = 0.0
        mujoco.mj_forward(self.model.ptr, self.data.ptr)

    def _limit(self, latch, active_normal, target, max_deflection):
        """Bound spring energy stored against a contacted surface.

        Once the target pushes into a contact, only the component continuing
        through it is capped; tangential motion and retreat are unaffected.
        """
        target = np.asarray(target, dtype=float)
        if latch[0] is not None:
            if active_normal is None:
                latch[2] += 1
                if latch[2] > CONTACT_GRACE_STEPS:
                    latch[:] = [None, None, 0]
            else:
                latch[2] = 0
        if latch[0] is None and active_normal is not None:
            if np.dot(target - self.tool_position, active_normal) < 0.0:
                latch[:] = [active_normal, float(np.dot(self.tool_position, active_normal)), 0]
        normal = latch[0]
        if normal is None:
            return target.copy()
        if float(np.dot(target, normal)) >= latch[1]:
            latch[:] = [None, None, 0]
            return target.copy()
        error = float(np.dot(target - self.tool_position, normal))
        if error >= -max_deflection:
            return target.copy()
        return target + (-max_deflection - error) * normal

    def _mean_normal(self, normals):
        if not normals:
            return None
        normal = np.mean(normals, axis=0)
        magnitude = np.linalg.norm(normal)
        return None if magnitude < 1e-9 else normal / magnitude

    def _surface_normal(self):
        """Mean outward normal of gripper contacts with the table or bookend."""
        normals = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if not self._guarded.intersection((int(contact.geom1), int(contact.geom2))):
                continue
            robot1 = int(self.model.geom_bodyid[contact.geom1]) in self._robot_bodies
            robot2 = int(self.model.geom_bodyid[contact.geom2]) in self._robot_bodies
            if robot1 == robot2:
                continue
            normal = np.asarray(contact.frame, dtype=float).reshape(3, 3)[0]
            normals.append(normal if robot2 else -normal)
        return self._mean_normal(normals)

    def _book_normal(self):
        """Mean outward normal of fingertip-pad contacts with the book."""
        normals = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 == self._book_geom and g2 in self._tip_geoms:
                tip_is_geom2 = True
            elif g2 == self._book_geom and g1 in self._tip_geoms:
                tip_is_geom2 = False
            else:
                continue
            normal = np.asarray(contact.frame, dtype=float).reshape(3, 3)[0]
            normals.append(normal if tip_is_geom2 else -normal)
        return self._mean_normal(normals)

    def _update_wrench(self, reset=False):
        """Exact contact wrench on the gripper, then a 30 Hz two-pole sensor filter."""
        total = np.zeros(6)
        origin = self.tool_position
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            robot1 = int(self.model.geom_bodyid[contact.geom1]) in self._robot_bodies
            robot2 = int(self.model.geom_bodyid[contact.geom2]) in self._robot_bodies
            if robot1 == robot2:
                continue
            mujoco.mj_contactForce(self.model.ptr, self.data.ptr, index, self._contact)
            to_world = np.asarray(contact.frame).reshape(3, 3).T
            force = to_world @ self._contact[:3]
            torque = to_world @ self._contact[3:]
            if not robot2:
                force, torque = -force, -torque
            torque += np.cross(np.asarray(contact.pos) - origin, force)
            total[:3] += force
            total[3:] += torque
        self._raw_wrench_world = total
        world_from_tool = np.asarray(self.data.site_xmat[self._tool_site]).reshape(3, 3)
        raw = np.concatenate([world_from_tool.T @ total[:3], world_from_tool.T @ total[3:]])
        if reset:
            self._filter = [raw.copy(), raw.copy()]
        else:
            self._filter[0] += self._filter_alpha * (raw - self._filter[0])
            self._filter[1] += self._filter_alpha * (self._filter[0] - self._filter[1])
        self._sensor_wrench = self._filter[1].copy()

    # ------------------------------------------------------------------- replay
    def get_state(self):
        state = np.empty(mujoco.mj_stateSize(self.model.ptr, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(
            self.model.ptr, self.data.ptr, state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        return state

    def close(self):
        self._camera = None
        self.physics.free()
