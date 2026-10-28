"""Sensor delivery, inference delay and plan updates for the DexMimicGen rollouts.

Observations are recorded at 100 Hz from the reset frame. Every observation
stream follows a persistent capture chain per (scene seed, timing group):
each capture event redraws its rate and delivery latency, is realized on the
next 100 Hz tick, and becomes visible once ``capture + latency <= query``. The
reset frame is always visible. A request sees the latest delivered frames,
ordered by capture time; earlier slots repeat the oldest frame (padding), and
repeated capture times keep only their last copy (duplicates). Even nominal
timing runs through this model (a fixed 20 Hz chain at phase zero).

Random variates are counter-based hashes of (seed, scene seed, group, event,
label), so every method sees the same timing in the same scene.

Policy calls may take a sampled inference delay ``D`` (whole 50 ms steps). The
fresh plan is queried at ``D + 0.05 j`` and executed from ``D`` steps after the
request; until then the newest completed plan keeps executing.
"""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

SOURCE_HZ = 100.0
STEP_SECONDS = 1.0 / 20.0
_EVENT_IDENTITY = "environment_seed_timing_group_delivery_seed_capture_index_and_label_sha256_v1"
_DELAY_IDENTITY = "environment_seed_and_policy_call_index_sha256_v1"


def _digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()


def counter_uniform(seed, scene_seed, group, index, label):
    payload = dict(
        capture_index=int(index),
        delivery_seed=int(seed),
        environment_seed=int(scene_seed),
        identity=_EVENT_IDENTITY,
        label=str(label),
        timing_group=str(group),
    )
    return float(np.random.default_rng(int.from_bytes(_digest(payload)[:16], "big")).random())


@dataclass(frozen=True)
class SensorTiming:
    """Per-capture rate (Hz) and latency (s) ranges; phase and stream grouping."""

    frequency_hz: tuple
    latency_s: tuple = (0.0, 0.0)
    random_phase: bool = True
    independent: bool = True  # separate chains for proprio and each camera

    def group(self, stream, is_image):
        if not self.independent:
            return "all_observations"
        return f"image:{stream}" if is_image else "proprio"

    def lookback_s(self, horizon):
        """Oldest capture age that can still be among the newest ``horizon`` deliveries."""
        return float(
            self.latency_s[1] + float(max(1, int(horizon))) / self.frequency_hz[0] + 1.0 / SOURCE_HZ
        )


NOMINAL = SensorTiming((20.0, 20.0), random_phase=False, independent=False)


class CaptureChain:
    """Lazily extended capture events of one timing group in one scene."""

    def __init__(self, timing, seed, scene_seed, group):
        self.timing, self.seed, self.scene_seed, self.group = timing, seed, scene_seed, group
        self.events = []  # (ideal, realized, period, latency)

    def _between(self, bounds, index, label):
        low, high = bounds
        value = float(
            low
            + counter_uniform(self.seed, self.scene_seed, self.group, index, label) * (high - low)
        )
        return float(low) if low == high else value

    def extend(self, query_s):
        """Generate events through ``query_s``; returns how many lie at or before it."""
        while not self.events or self.events[-1][0] <= query_s + 1e-12:
            index = len(self.events)
            period = 1.0 / self._between(self.timing.frequency_hz, index, "frequency")
            latency = self._between(self.timing.latency_s, index, "delivery_latency")
            if index == 0:
                phase = counter_uniform(self.seed, self.scene_seed, self.group, 0, "phase_fraction")
                ideal = float(phase * period) if self.timing.random_phase else 0.0
            else:
                ideal = float(self.events[-1][0] + self.events[-1][2])
            realized = float(int(math.ceil(ideal * SOURCE_HZ - 1e-10))) / SOURCE_HZ
            self.events.append((ideal, realized, period, latency))
        return sum(1 for event in self.events if event[0] <= query_s + 1e-12)


def _duplicates(times, padding):
    """All but the last non-padding copy of each repeated capture time."""
    duplicate = np.zeros(len(times), dtype=np.bool_)
    positions = {}
    for position, time in enumerate(times):
        positions.setdefault(float(time), []).append(position)
    for group in positions.values():
        kept = [p for p in group if not padding[p]]
        duplicate[group] = True
        duplicate[kept[-1] if kept else group[-1]] = False
    return duplicate


@dataclass(frozen=True)
class Selection:
    indices: np.ndarray  # into the retained source window
    capture_s: np.ndarray  # realized capture times relative to the request
    padding: np.ndarray
    duplicate: np.ndarray


def select_delivered(chain, source_s, horizon):
    """Latest ``horizon`` frames delivered by the newest source time."""
    source_s = np.asarray(source_s, dtype=np.float64)
    query = float(source_s[-1])
    end = chain.extend(query)
    oldest = max(0.0, query - chain.timing.lookback_s(horizon))
    start = next((i for i in range(end) if chain.events[i][0] >= oldest - 1e-12), end)
    candidates = [(0.0, -1)]  # the reset frame needs no delivery
    for index in range(start, end):
        _, realized, _, latency = chain.events[index]
        if realized > 1e-12 and realized + latency <= query + 1e-10:
            candidates.append((realized, index))
    candidates.sort()
    unique = []
    for candidate in candidates:
        if unique and np.isclose(candidate[0], unique[-1][0], rtol=0.0, atol=1e-12):
            unique[-1] = candidate  # same source tick: keep the later event
        else:
            unique.append(candidate)
    selected = unique[-horizon:]
    padding = np.zeros(horizon, dtype=np.bool_)
    padding[: horizon - len(selected)] = True
    selected = [selected[0]] * (horizon - len(selected)) + selected
    captured = np.array([c[0] for c in selected], dtype=np.float64)
    indices = []
    for time in captured:
        right = int(np.searchsorted(source_s, time, side="left"))
        frame = min(
            (i for i in (right - 1, right) if 0 <= i < source_s.size),
            key=lambda i: abs(float(source_s[i]) - time),
        )
        if abs(float(source_s[frame]) - time) > max(1e-9, 1e-7 / SOURCE_HZ):
            raise RuntimeError(f"No retained frame at capture time {time:.12g} s")
        indices.append(frame)
    relative = captured - query
    return Selection(
        np.array(indices, dtype=np.int64), relative, padding, _duplicates(relative, padding)
    )


class DeliveredSensors:
    """Per-episode capture chains, shared by all streams of one timing group."""

    def __init__(self, timing, seed, scene_seed):
        self.timing, self.seed, self.scene_seed = timing, seed, scene_seed
        self.chains = {}

    def select(self, stream, is_image, source_s, horizon):
        group = self.timing.group(stream, is_image)
        if group not in self.chains:
            self.chains[group] = CaptureChain(self.timing, self.seed, self.scene_seed, group)
        return select_delivered(self.chains[group], source_s, horizon)


def inference_delay_s(seed, scene_seed, call, maximum_s, quantum_s=0.05):
    """Uniform delay in {0, 50, ..., maximum} ms; call 0 is never delayed."""
    if call == 0 or not maximum_s:
        return 0.0
    payload = dict(
        base_seed=int(seed),
        environment_seed=int(scene_seed),
        identity=_DELAY_IDENTITY,
        policy_call_index=int(call),
    )
    u = float(int.from_bytes(_digest(payload)[:8], "big")) / float(2**64)
    count = int(round(float(maximum_s) / quantum_s)) + 1
    return float(round(float(min(int(math.floor(u * count)), count - 1)) * quantum_s, 12))


class PlanSchedule:
    """Asynchronous plan updates over ``execute_steps``-step calls.

    A call's schedule is the unconsumed head of the previous plan for the
    delay, then the fresh plan. The first call has no completed plan and holds
    the controllers' reset targets during any delay.
    """

    def __init__(self, execute_steps):
        self.execute_steps = int(execute_steps)
        self.plan = None
        self.cursor = 0

    def update(self, fresh, delay_s, previous_action):
        delay = int(np.rint(delay_s / STEP_SECONDS))
        if not 0 <= delay < self.execute_steps:
            raise ValueError("The inference delay must be shorter than the executed chunk")
        fresh = np.asarray(fresh, dtype=np.float32)
        head = fresh[: self.execute_steps - delay]
        if self.plan is None:
            schedule = dict(
                actions=head,
                latency=delay * STEP_SECONDS,
                hold=np.zeros(fresh.shape[1], np.float32),
            )
        else:
            if self.cursor + delay > len(self.plan):
                raise RuntimeError("The inference delay exhausted the previous plan")
            actions = np.concatenate((self.plan[self.cursor : self.cursor + delay], head), axis=0)
            schedule = dict(
                actions=actions, latency=0.0, hold=np.asarray(previous_action, np.float32)
            )
        self.plan = fresh.copy()
        self.cursor = self.execute_steps - delay
        schedule["index"] = None
        return schedule
