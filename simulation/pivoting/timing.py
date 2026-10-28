"""Sensor delivery and delayed action-plan updates on a 1 ms simulation clock.

Each sensor stream captures at its own rate and arrives after a delivery
latency; the policy sees only delivered samples. Policy calls may also take
a sampled inference delay, during which the previous plan keeps executing.
All randomness is derived from (seed, episode, stream), so every method sees
the same timing in the same episode.
"""

from dataclasses import dataclass, field
import hashlib
import heapq
import math

import numpy as np


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class SensorTiming:
    """Capture rates (Hz, sampled uniformly per capture), latency and phase per stream.

    ``first_capture_ms`` fixes each stream's first capture tick; when omitted,
    fixed-rate streams first capture after one interval and jittered streams
    at a random phase within their first interval.
    """

    rates_hz: dict
    latency_ms: int = 0
    first_capture_ms: dict = field(default=None)

    @property
    def nominal(self):
        fixed = all(low == high for low, high in self.rates_hz.values())
        return fixed and self.latency_ms == 0 and self.first_capture_ms is None


def inference_delay_ms(episode, call, maximum_ms, seed, quantum_ms=50):
    """Uniform delay in {0, 50, ..., maximum} ms; the first call is never delayed."""
    if call == 0 or maximum_ms == 0:
        return 0
    rng = np.random.default_rng(stable_seed("inference", seed, episode, call))
    return quantum_ms * int(rng.integers(0, maximum_ms // quantum_ms + 1))


class DeliveredSensors:
    """Buffers captured samples and releases them when their latency has elapsed."""

    def __init__(self, timing, horizons, episode, initial, seed):
        self.timing, self.horizons = timing, horizons
        self.pending = []
        self.next = {}
        self.counter = {name: 0 for name in horizons}
        self.frequency_rng, self.latency_rng = {}, {}
        # Every stream starts with the same initial sample for both methods.
        self.delivered = {
            name: [dict(source_tick=0, capture_tick=0, delivery_tick=0, value=initial[name])]
            for name in horizons
        }
        for name in horizons:
            self.frequency_rng[name] = np.random.default_rng(
                stable_seed("frequency", seed, episode, name)
            )
            self.latency_rng[name] = np.random.default_rng(
                stable_seed("sensor_latency", seed, episode, name)
            )
            interval = self._interval(name)
            if timing.nominal:
                self.next[name] = interval
            elif timing.first_capture_ms is not None:
                self.next[name] = timing.first_capture_ms[name]
            else:
                phase = np.random.default_rng(stable_seed("phase", seed, episode, name))
                self.next[name] = int(phase.integers(1, interval + 1))

    def _interval(self, name):
        low, high = self.timing.rates_hz[name]
        frequency = low if low == high else self.frequency_rng[name].uniform(low, high)
        return int(math.ceil(1000.0 / frequency - 1e-9))

    def update(self, tick, latest):
        """Capture streams due at ``tick`` from ``latest[name] = (source_tick, value)``."""
        for name in self.horizons:
            if tick < self.next[name]:
                continue
            assert tick == self.next[name]
            self.counter[name] += 1
            latency = (
                0
                if self.timing.nominal
                else int(self.latency_rng[name].integers(0, self.timing.latency_ms + 1))
            )
            source_tick, value = latest[name]
            packet = dict(
                source_tick=source_tick,
                capture_tick=tick,
                delivery_tick=tick + latency,
                value=value,
            )
            heapq.heappush(self.pending, (tick + latency, name, self.counter[name], packet))
            self.next[name] = tick + self._interval(name)
        while self.pending and self.pending[0][0] <= tick:
            _, name, _, packet = heapq.heappop(self.pending)
            self.delivered[name].append(packet)

    def observation(self, name, tick):
        """Latest ``horizon`` delivered samples, capture times (s, relative) and validity.

        Missing history repeats the oldest sample and is marked invalid, as are
        all but the last of samples sharing a capture time.
        """
        available = sorted(self.delivered[name], key=lambda p: p["capture_tick"])
        count = self.horizons[name]
        selected = available[-count:]
        padding = [False] * len(selected)
        if len(selected) < count:
            missing = count - len(selected)
            selected = [selected[0]] * missing + selected
            padding = [True] * missing + padding
        clocks = np.array([(p["source_tick"] - tick) * 0.001 for p in selected], dtype=np.float64)
        duplicate = np.zeros(count, dtype=bool)
        for value in np.unique(clocks):
            ids = np.flatnonzero(clocks == value)
            keep = ([i for i in ids if not padding[i]] or list(ids))[-1]
            duplicate[ids] = True
            duplicate[keep] = False
        valid = ~(np.array(padding) | duplicate)
        return np.stack([p["value"] for p in selected]), clocks, valid


def schedule_update(fresh, previous, request_tick, delay_ms, duration_ms):
    """Targets for the next ``duration_ms`` ticks after a policy call.

    The previous plan keeps running for the inference delay; the fresh plan,
    whose time origin is the end of the delay, takes over atomically.
    """
    assert fresh.origin_tick == request_tick + delay_ms and 0 <= delay_ms < duration_ms
    ticks = np.arange(request_tick + 1, request_tick + duration_ms + 1, dtype=np.int64)
    if not delay_ms:
        return fresh.sample(ticks)
    if previous is None:
        raise ValueError("The first policy call must not be delayed")
    prefix = previous.sample(ticks[:delay_ms])
    suffix = fresh.sample(ticks[delay_ms:])
    return tuple(np.concatenate([a, b]) for a, b in zip(prefix, suffix))
