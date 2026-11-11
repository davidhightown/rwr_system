"""When may a run start or continue, given the motor temperatures? No ROS.

A run (or the rest of it) may start when no motor is expected to reach the limit before it ends:

    temperature + heating rate * remaining seconds + margin <= limit      for every motor

The heating rate of each motor (deg C per second of policy time) is learned from earlier runs: the largest rate of
its last runs, stored in a JSON file. Without enough runs, a default rate is used. A motor heats more slowly the
hotter it is (it loses more heat), so a rate measured from a cool start is on the safe side.
"""

import json
from pathlib import Path

import numpy as np

DEFAULT_RATE = 0.25   # deg C/s: 15 deg C in 60 s, until a motor has MIN_RUNS measured runs
MIN_RUNS = 3
KEEP_RUNS = 8         # the rate is the largest of this many latest runs
MIN_SEGMENT_S = 10.0  # shorter stretches of policy time are too noisy for a rate (readings are whole degrees)


class TemperatureFilter:
    """The servos sometimes answer a wrong temperature. A reading that jumps by more than ``max_jump`` from the
    last accepted one, or lies outside 0..100 deg C, is ignored unless the next reading confirms it."""

    def __init__(self, max_jump=10.0):
        self.max_jump = max_jump
        self.values = None
        self.pending = None
        self.glitches = 0

    def update(self, reading):
        reading = np.asarray(reading, dtype=float)
        if self.values is None or self.values.shape != reading.shape:
            ok = np.isfinite(reading) & (reading > 0) & (reading < 100)
            self.values = np.where(ok, reading, np.nan)
            self.pending = np.full(reading.shape, np.nan)
            return self.values
        impossible = ~np.isfinite(reading) | (reading <= 0) | (reading >= 100)
        jumped = np.abs(reading - self.values) > self.max_jump
        confirmed = np.abs(reading - self.pending) <= 3.0
        reject = impossible | (jumped & ~confirmed & np.isfinite(self.values))
        self.pending = np.where(reject & ~impossible, reading, np.nan)
        self.glitches += int(reject.sum())
        self.values = np.where(reject, self.values, reading)
        return self.values


class ThermalModel:
    def __init__(self, joint_names, limit=60.0, margin=2.0, history_path=None):
        self.joint_names = list(joint_names)
        self.limit, self.margin = float(limit), float(margin)
        self.history_path = None if history_path is None else Path(history_path)
        self.history = {}  # joint -> list of rates (deg C/s), oldest first
        if self.history_path is not None and self.history_path.exists():
            try:
                self.history = {k: [float(v) for v in vals] for k, vals in json.loads(self.history_path.read_text()).items()}
            except (ValueError, AttributeError, TypeError):
                self.history = {}

    def rates(self):
        """Expected heating rate per joint (deg C/s)."""
        out = []
        for joint in self.joint_names:
            runs = self.history.get(joint, [])[-KEEP_RUNS:]
            out.append(max(runs) if len(runs) >= MIN_RUNS else max([DEFAULT_RATE] + runs))
        return np.array(out)

    def headroom(self, temps, remaining_s):
        """Per joint: deg C left below the limit at the end of the remaining time (negative: too hot to start)."""
        temps = np.asarray(temps, dtype=float)
        return self.limit - self.margin - (temps + self.rates() * max(0.0, remaining_s))

    def ready(self, temps, remaining_s):
        headroom = self.headroom(temps, remaining_s)
        return bool(np.all(np.isfinite(headroom)) and np.all(headroom >= 0))

    def start_below(self, remaining_s):
        """Per joint: the temperature below which the remaining time may start."""
        return self.limit - self.margin - self.rates() * max(0.0, remaining_s)

    def over_limit(self, temps):
        """Index of the hottest joint at or above the limit, else None."""
        temps = np.asarray(temps, dtype=float)
        if not np.any(np.isfinite(temps)):
            return None
        hottest = int(np.nanargmax(temps))
        return hottest if temps[hottest] >= self.limit else None

    def learn(self, segments):
        """Add a run. segments: [(policy seconds, temperatures at its start, temperatures at its end)], one per
        uninterrupted stretch of policy time. The run's rate per joint is the largest of its long enough segments."""
        rates = [np.maximum(0.0, (np.asarray(end, float) - np.asarray(start, float)) / seconds)
                 for seconds, start, end in segments if seconds >= MIN_SEGMENT_S]
        rates = [r for r in rates if np.all(np.isfinite(r))]
        if not rates:
            return None
        run_rate = np.max(rates, axis=0)
        for joint, rate in zip(self.joint_names, run_rate):
            self.history[joint] = (self.history.get(joint, []) + [round(float(rate), 4)])[-KEEP_RUNS:]
        if self.history_path is not None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(self.history, indent=1))
        return run_rate
