"""DexMimicGen bimanual tasks on the official robosuite / robomimic environments.

The environment is rebuilt from the dataset's recorded robomimic metadata,
with the controllers switched to absolute end-effector targets and the control
clock set to the 20 Hz policy rate. Each 20 Hz step runs 25 MuJoCo steps of
2 ms through robosuite's own step1 / _pre_action / step2 / _update_observables
sequence, so an action target can change at any physics step; observations
are recorded every 10 ms (a forced sensor update), giving the 100 Hz record
that sensor delivery selects from.

Episodes always last ``max_steps`` policy steps in the paper; success is the
maximum sparse reward reaching 1, so ``step`` reports success as soon as the
task is first solved.
"""

import copy
import json
import math
import os
import random
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import zarr

POLICY_HZ = 20
SOURCE_HZ = 100  # observation record
STEP_SECONDS = 1.0 / POLICY_HZ

# Logical policy streams per task: (policy key, position, quaternion, gripper, gripper width used).
STANDARD_PROPRIO = (
    ("robot0_10d", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", 1),
    ("robot1_10d", "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", 1),
)
GR1_PROPRIO = (  # GR1 humanoid: one robot with right and left arms
    (
        "robot0_20d",
        "robot0_right_eef_pos",
        "robot0_right_eef_quat",
        "robot0_right_gripper_qpos",
        11,
    ),
    ("robot1_20d", "robot0_left_eef_pos", "robot0_left_eef_quat", "robot0_left_gripper_qpos", 11),
)
DEX_PROPRIO = (  # Panda arms with dexterous hands
    ("robot0_21d", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", 12),
    ("robot1_21d", "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", 12),
)
# Raw action layout per logical stream: (policy key, pose slice, hand slice). The
# pose is xyz + axis-angle; GR1's raw order is [right pose, left pose, right hand, left hand].
TASKS = {
    "threading": dict(proprio=STANDARD_PROPRIO, actions=None),
    "assembly": dict(proprio=STANDARD_PROPRIO, actions=None),
    "transport": dict(
        proprio=STANDARD_PROPRIO,
        actions=(("action0_10d", (0, 6), (6, 7)), ("action1_10d", (7, 13), (13, 14))),
    ),
    "drawer_cleanup": dict(
        proprio=DEX_PROPRIO,
        actions=(("action0_15d", (0, 6), (6, 12)), ("action1_15d", (12, 18), (18, 24))),
    ),
    "coffee": dict(
        proprio=GR1_PROPRIO,
        actions=(("action0_15d", (0, 6), (12, 18)), ("action1_15d", (6, 12), (18, 24))),
    ),
    "pouring": dict(
        proprio=GR1_PROPRIO,
        actions=(("action0_15d", (0, 6), (12, 18)), ("action1_15d", (6, 12), (18, 24))),
    ),
}
WHOLE_BODY_IK = ("WHOLE_BODY_MINK_IK", "WHOLE_BODY_IK")


# ------------------------------------------------------------- metadata
def load_env_meta(data_dir):
    """robomimic env metadata: the store's ``env_args``, else ``evaluation.hdf5``."""
    data_dir = Path(data_dir)
    attrs = zarr.open(str(data_dir / "episode_data.zarr"), "r").attrs
    raw = attrs.get("env_args", attrs.get("robomimic_env_args"))
    if raw is None:
        with h5py.File(data_dir / "evaluation.hdf5", "r") as f:
            raw = f["data"].attrs["env_args"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return copy.deepcopy(json.loads(raw) if isinstance(raw, str) else dict(raw))


def stored_image_size(data_dir, camera):
    """The dataset's stored image height and width; cameras render at this size."""
    store = zarr.open(str(Path(data_dir) / "episode_data.zarr"), "r")
    names = sorted(
        store.group_keys(),
        key=lambda n: (0, int(n.rsplit("_", 1)[-1])) if n.rsplit("_", 1)[-1].isdigit() else (1, n),
    )
    for name in names:
        if camera in store[name]:
            return tuple(int(d) for d in store[name][camera].shape[-3:-1])
    raise KeyError(f"{camera} is not stored in {data_dir}")


def configure_env_meta(meta, image_size):
    """Absolute end-effector control at the policy rate, cameras at the stored size."""
    kwargs = meta.setdefault("env_kwargs", {})
    kwargs.pop("lite_physics", None)
    kwargs.pop("env_lang", None)
    controller = kwargs.get("controller_configs")
    if controller.get("type") == "BASIC":
        for part in controller["body_parts"].values():
            if "damping" in part and "damping_ratio" not in part:
                part["damping_ratio"] = part.pop("damping")
            if "damping_limits" in part and "damping_ratio_limits" not in part:
                part["damping_ratio_limits"] = part.pop("damping_limits")
            if part.get("type") != "GRIP":
                part["control_delta"] = False
                part["input_type"] = "absolute"
    elif controller.get("type") in WHOLE_BODY_IK:
        specific = controller.setdefault("composite_controller_specific_configs", {})
        if specific.get("ik_input_type") != "absolute":
            raise ValueError("The whole-body IK controller must take absolute targets")
        specific["ik_solve_freq"] = float(POLICY_HZ)
    else:
        raise ValueError(f"Unsupported controller {controller.get('type')!r}")
    kwargs["control_freq"] = POLICY_HZ
    kwargs["camera_heights"], kwargs["camera_widths"] = image_size
    kwargs["use_object_obs"] = False
    return meta


# --------------------------------------------------------------- actions
def rot6d_matrix(rot6d):
    """First two rotation-matrix rows -> matrix (Gram-Schmidt), float64."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a = rot6d[..., :3] / np.maximum(np.linalg.norm(rot6d[..., :3], axis=-1, keepdims=True), 1e-12)
    b = rot6d[..., 3:] - np.sum(a * rot6d[..., 3:], axis=-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return np.stack((a, b, np.cross(a, b, axis=-1)), axis=-2)


def action_keys(task):
    layout = TASKS[task]["actions"]
    return ("action0_10d", "action1_10d") if layout is None else tuple(spec[0] for spec in layout)


def to_env_actions(task, actions):
    """Policy actions [T, D] (xyz, rot6d, hand per stream, in ``action_keys`` order) -> raw env actions."""
    actions = np.asarray(actions, dtype=np.float32)
    layout = TASKS[task]["actions"]
    if layout is None:
        # Two 10-D arms, converted as the robomimic runner does (float32 rows).
        arms = actions.reshape(-1, 2, 10)
        first, second = arms[..., 3:6].reshape(-1, 3), arms[..., 6:9].reshape(-1, 3)
        first = first / np.linalg.norm(first, axis=-1, keepdims=True).clip(1e-12)
        second = second - (first * second).sum(-1, keepdims=True) * first
        second = second / np.linalg.norm(second, axis=-1, keepdims=True).clip(1e-12)
        rotvec = Rotation.from_matrix(
            np.stack((first, second, np.cross(first, second)), axis=-2)
        ).as_rotvec()
        raw = np.concatenate(
            [arms[..., :3], rotvec.reshape(arms.shape[:-1] + (3,)), arms[..., -1:]], axis=-1
        )
        return raw.reshape(actions.shape[0], 14).astype(np.float32)
    raw = np.zeros((actions.shape[0], max(s[2][1] for s in layout)), dtype=np.float32)
    start = 0
    for _, (p0, p1), (h0, h1) in layout:
        stream = actions[:, start : start + 9 + h1 - h0]
        start += 9 + h1 - h0
        rotvec = Rotation.from_matrix(rot6d_matrix(stream[:, 3:9])).as_rotvec().astype(np.float32)
        raw[:, p0:p1] = np.concatenate([stream[:, :3], rotvec], axis=-1)
        raw[:, h0:h1] = stream[:, 9:]
    return raw


def scheduled_action(schedule, t):
    """Target at ``t`` seconds into a call; index -1 while holding during the delay."""
    if t < schedule["latency"] - 1e-9:
        return schedule["hold"], -1
    index = min(
        math.floor((t - schedule["latency"]) / STEP_SECONDS + 1e-9), len(schedule["actions"]) - 1
    )
    return schedule["actions"][index], index


# ------------------------------------------------------------------- env
class DexMimicGenEnv:
    """One robomimic ``EnvRobosuite`` stepped on the MuJoCo clock with a 100 Hz record."""

    def __init__(self, task, data_dir, cameras, max_steps):
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        import dexmimicgen.environments  # noqa: F401  (registers the tasks)

        self.task, self.cameras, self.max_steps = task, tuple(cameras), int(max_steps)
        self.proprio = TASKS[task]["proprio"]
        self.low_dim_keys = tuple(k for spec in self.proprio for k in spec[1:4])
        self.keys = self.cameras + self.low_dim_keys
        self.image_size = stored_image_size(data_dir, self.cameras[0])
        meta = configure_env_meta(load_env_meta(data_dir), self.image_size)
        device = os.environ.get("EGL_DEVICE_ID", "").strip()
        if device.isdigit():
            import egl_probe

            egl_probe.get_available_devices = lambda: [int(device)]
        ObsUtils.initialize_obs_modality_mapping_from_dict(
            {"rgb": list(self.cameras), "low_dim": list(self.low_dim_keys)}
        )
        self.env = EnvUtils.create_env_from_metadata(
            env_meta=meta, render=False, render_offscreen=True, use_image_obs=True
        )
        self.sim_env = self.env.env
        self.sim_env.hard_reset = False
        self.substeps = int(round(STEP_SECONDS / self.sim_env.model_timestep))

    def _observe(self, raw):
        obs = self.env.get_observation(raw)
        return {key: np.array(obs[key], copy=True) for key in self.keys}

    def reset(self, seed):
        """Seeded reset that rebuilds the model; returns the t = 0 observation."""
        np.random.seed(seed)
        random.seed(seed)
        self.sim_env.hard_reset = True
        try:
            np.random.seed(seed)
            obs = self.env.reset()
        finally:
            self.sim_env.hard_reset = False
        self.steps, self.time, self.success = 0, 0.0, False
        self.last_capture = 0.0
        return dict(time=0.0, **{key: np.array(obs[key], copy=True) for key in self.keys})

    @property
    def done(self):
        return self.success or self.steps >= self.max_steps

    def step(self, schedule, start_s):
        """One 20 Hz step of a call's schedule beginning ``start_s`` into the call.

        Returns the five 100 Hz observations recorded during the step. A new
        target reaches the controllers only when the scheduled index changes.
        """
        env = self.sim_env
        dt = env.model_timestep
        env.timestep += 1
        frames, final = [], False
        action = None
        for i in range(self.substeps):
            action, index = scheduled_action(schedule, start_s + i * dt)
            policy_step = index >= 0 and (schedule["index"] is None or index != schedule["index"])
            if env.lite_physics:
                env.sim.step1()
            else:
                env.sim.forward()
            env._pre_action(action, policy_step)
            if env.lite_physics:
                env.sim.step2()
            else:
                env.sim.step()
            before = env.cur_time + i * dt
            crossed = math.floor((before + dt) * SOURCE_HZ + 1e-9) > math.floor(
                before * SOURCE_HZ + 1e-9
            )
            env._update_observables(force=crossed)
            if crossed and i == self.substeps - 1:
                final = True
            elif crossed:
                frames.append(self._observe(env._get_observations()))
            schedule["index"] = index
        env.cur_time += self.substeps * dt
        reward, _, _ = env._post_action(action)
        observation = env._get_observations()
        if final:
            frames.append(self._observe(observation))
        # Capture times continue from the previous capture, rounded below float noise.
        records = [
            dict(time=round(self.last_capture + (k + 1) * (1.0 / SOURCE_HZ), 12), **frame)
            for k, frame in enumerate(frames)
        ]
        if records:
            self.last_capture = records[-1]["time"]
        self.time = round(self.time + STEP_SECONDS, 12)
        self.steps += 1
        self.success = self.success or float(reward) >= 1.0
        return records

    def close(self):
        self.env.env.close()
