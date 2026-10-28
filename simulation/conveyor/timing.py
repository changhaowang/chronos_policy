"""Sensor delivery, inference delay and plan updates on the conveyor's 100 Hz clock.

Observations are recorded at 100 Hz from episode time zero (the reset frame).
A policy request at time ``q`` sees, per stream, the latest two frames that
were *delivered* by ``q``:

* nominal timing selects the frames nearest to ``q - 0.05 k`` (k = 1, 0),
  i.e. the 20 Hz control grid;
* randomized timing runs a capture chain per (scene seed, stream). Each
  capture event redraws its rate and delivery latency, is realized on the
  next 100 Hz source tick, and becomes visible once ``capture + latency <= q``.
  The reset frame is always visible.

Every random variate is a counter-based hash of (seed, scene seed, stream,
event index, label), so timing does not depend on access order and every
method sees the same timing in the same scene. Capture times are reported
relative to the request. Slots repeating an earlier frame are padding (before
the reset frame) or duplicates (a frame selected twice).

Policy calls may take a sampled inference delay ``D`` (whole 50 ms action
periods). The fresh plan is queried at ``D + 0.05 j`` and lands ``D`` steps
after the request; meanwhile the unconsumed part of the previous plan keeps
executing.
"""

from dataclasses import dataclass
import copy
import hashlib
import json
import math

import numpy as np

SOURCE_HZ = 100.0
ACTION_PERIOD_S = 1.0 / 20.0
TIME_ATOL_S = 1e-10
_EVENT_IDENTITY = "environment_seed_timing_group_delivery_seed_capture_index_and_label_sha256_v1"
_DELAY_IDENTITY = "environment_seed_and_policy_call_index_sha256_v1"


def _digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()


def counter_uniform(seed, scene_seed, stream, index, label):
    """U[0, 1) keyed by its arguments only."""
    payload = dict(
        capture_index=int(index),
        delivery_seed=int(seed),
        environment_seed=int(scene_seed),
        identity=_EVENT_IDENTITY,
        label=str(label),
        timing_group=str(stream),
    )
    return float(np.random.default_rng(int.from_bytes(_digest(payload)[:16], "big")).random())


def _between(low, high, u):
    return float(low) if low == high else float(low + u * (high - low))


@dataclass(frozen=True)
class SensorTiming:
    """Per-capture rate (Hz) and delivery latency (s) ranges, sampled uniformly."""

    frequency_hz: tuple
    latency_s: tuple = (0.0, 0.0)
    random_phase: bool = True  # first capture at a random phase of its period


@dataclass
class Event:
    ideal_s: float
    period_s: float
    latency_s: float


class CaptureChain:
    """Lazily extended capture events of one stream in one scene."""

    def __init__(self, timing, seed, scene_seed, stream):
        self.timing, self.seed, self.scene_seed, self.stream = timing, seed, scene_seed, stream
        self.events = []

    def _uniform(self, index, label):
        return counter_uniform(self.seed, self.scene_seed, self.stream, index, label)

    def extend(self, query_s):
        """Generate events until one lies after ``query_s``."""
        low, high = self.timing.frequency_hz
        while not self.events or self.events[-1].ideal_s <= query_s + TIME_ATOL_S:
            index = len(self.events)
            period = 1.0 / _between(low, high, self._uniform(index, "frequency"))
            if index == 0:
                ideal = (
                    self._uniform(0, "phase_fraction") * period if self.timing.random_phase else 0.0
                )
            else:
                ideal = self.events[-1].ideal_s + self.events[-1].period_s
            latency = _between(*self.timing.latency_s, self._uniform(index, "delivery_latency"))
            self.events.append(Event(float(ideal), float(period), latency))


@dataclass(frozen=True)
class Selection:
    indices: np.ndarray  # into the source clock
    capture_s: np.ndarray  # absolute source times of the selected frames
    padding: np.ndarray
    duplicate: np.ndarray

    @property
    def valid(self):
        return ~(self.padding | self.duplicate)


def _duplicates(indices, padding):
    """All but one copy of a repeated frame; the kept copy is its last non-padding slot."""
    duplicate = np.zeros(len(indices), dtype=np.bool_)
    positions = {}
    for position, index in enumerate(indices.tolist()):
        positions.setdefault(index, []).append(position)
    for group in positions.values():
        kept = [p for p in group if not padding[p]]
        keep = kept[-1] if kept else group[-1]
        for p in group:
            duplicate[p] = p != keep
    return duplicate


def select_delivered(chain, source_s, query_s, horizon):
    """Latest ``horizon`` frames delivered by ``query_s``, ordered by capture time."""
    source_s = np.asarray(source_s, dtype=np.float64)
    candidates = [(0.0, 0, -1)]  # the reset frame needs no delivery
    chain.extend(query_s)
    events = chain.events
    end = len(events)
    while end > 0 and events[end - 1].ideal_s > query_s + TIME_ATOL_S:
        end -= 1
    # Only events recent enough to be among the newest ``horizon`` deliveries.
    oldest = max(
        0.0,
        query_s
        - chain.timing.latency_s[1]
        - (horizon + 2) * (1.0 / chain.timing.frequency_hz[0])
        - 1.0 / SOURCE_HZ,
    )
    start, high = 0, end
    while start < high:
        middle = (start + high) // 2
        if events[middle].ideal_s < oldest - TIME_ATOL_S:
            start = middle + 1
        else:
            high = middle
    for index in range(start, end):
        event = events[index]
        captured = float(int(math.ceil(event.ideal_s * SOURCE_HZ - 1e-10))) / SOURCE_HZ
        if captured <= TIME_ATOL_S:
            continue  # coincides with the reset frame
        right = int(np.searchsorted(source_s, captured, side="left"))
        nearby = [i for i in (right - 1, right) if 0 <= i < source_s.size]
        frame = min(nearby, key=lambda i: abs(float(source_s[i]) - captured))
        if abs(float(source_s[frame]) - captured) > max(1e-9, 1e-7 / SOURCE_HZ):
            raise RuntimeError(f"No recorded frame at capture time {captured:.12g} s")
        if captured + event.latency_s <= query_s + TIME_ATOL_S:
            candidates.append((captured, frame, index))
    candidates.sort(key=lambda c: (c[0], c[2]))
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
    indices = np.asarray([c[1] for c in selected], dtype=np.int64)
    return Selection(indices, source_s[indices], padding, _duplicates(indices, padding))


def select_nominal(source_s, query_s, horizon, step_s=ACTION_PERIOD_S):
    """Frames nearest to the control grid ``query - step * k``, clipped at time zero."""
    source = np.asarray(source_s, dtype=np.float64).reshape(-1)
    requested = float(query_s) - step_s * np.arange(horizon - 1, -1, -1, dtype=np.float64)
    padding = requested < source[0] - 1e-12
    clipped = np.clip(requested, source[0], source[-1])
    right = np.clip(np.searchsorted(source, clipped, side="left"), 0, source.size - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(
        np.abs(source[left] - clipped) <= np.abs(source[right] - clipped), left, right
    ).astype(np.int64)
    return Selection(indices, source[indices], padding, _duplicates(indices, padding))


class DeliveredSensors:
    """Per-episode stream selections; ``timing=None`` selects the nominal grid."""

    def __init__(self, timing, seed, scene_seed):
        self.timing, self.seed, self.scene_seed = timing, seed, scene_seed
        self.chains = {}

    def select(self, stream, source_s, query_s, horizon):
        if self.timing is None:
            return select_nominal(source_s, query_s, horizon)
        if stream not in self.chains:
            self.chains[stream] = CaptureChain(self.timing, self.seed, self.scene_seed, stream)
        return select_delivered(self.chains[stream], source_s, query_s, horizon)


def inference_delay_steps(seed, scene_seed, call, maximum_s, quantum_s=ACTION_PERIOD_S):
    """Uniform delay in {0, 1, ..., maximum / 50 ms} action periods; call 0 is never delayed."""
    if call == 0 or not maximum_s:
        return 0
    payload = dict(
        base_seed=int(seed),
        environment_seed=int(scene_seed),
        identity=_DELAY_IDENTITY,
        policy_call_index=int(call),
    )
    u = float(int.from_bytes(_digest(payload)[:8], "big")) / float(2**64)
    count = int(round(float(maximum_s) / quantum_s)) + 1
    delay_s = float(min(int(np.floor(u * count)), count - 1)) * quantum_s
    return int(round(delay_s / ACTION_PERIOD_S))


class PlanSchedule:
    """Asynchronous plan updates: the newest completed plan covers inference delays."""

    def __init__(self, horizon, execute_steps):
        self.horizon, self.execute_steps = horizon, execute_steps
        self.plan = None
        self.cursor = 0

    def update(self, fresh, delay_steps, previous_action):
        """Actions for the steps after a request whose plan arrives ``delay_steps`` late.

        ``fresh`` holds the new plan (at least ``horizon - delay_steps``
        actions). The first request has no completed plan and holds the reset
        command instead.
        """
        if self.plan is None:
            prefix = [copy.deepcopy(previous_action) for _ in range(delay_steps)]
        else:
            if self.cursor + delay_steps > len(self.plan):
                raise RuntimeError("The inference delay exhausted the previous plan")
            prefix = copy.deepcopy(self.plan[self.cursor : self.cursor + delay_steps])
        actions = prefix + list(fresh[: self.horizon - delay_steps])
        self.plan = copy.deepcopy(actions[delay_steps:])
        self.cursor = max(0, self.execute_steps - delay_steps)
        return actions
