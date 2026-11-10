"""Turn counting and drop detection from a stream of detected cube poses (camera frame), without ROS.

The detector's orientation of a cube is only defined up to the cube's 24 rotational symmetries, and single detections
can be wrong (e.g. a PnP flip). The counter therefore

- matches each new orientation to the symmetric one closest to the last accepted orientation,
- rejects steps that the cube cannot make between two detections (more than max_step about the vertical camera axis or
  more than max_tilt about the other axes), and re-anchors after a few rejected detections in a row,
- keeps every accepted step as a rotation vector, so that the turns can be counted about the actual spin axis after the
  run. The camera axis only sees the projection of a tilted spin axis: cos(tilt) of the turns.

Rotation during a gap without detections is only counted correctly up to 45 degrees.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

CUBE_SYMMETRIES = Rotation.create_group("O")
REANCHOR_AFTER = 5  # rejected detections in a row


@dataclass
class Detection:
    time: float
    position: np.ndarray  # m, camera frame
    rotation: Rotation  # cube orientation, camera frame (the symmetric one closest to the previous detection)
    accepted: bool
    step: np.ndarray  # rotation vector from the previous accepted orientation (rad, camera frame)


@dataclass
class TurnResult:
    turns: float  # about the estimated spin axis, positive = counterclockwise seen from above
    turns_camera_axis: float
    spin_axis_tilt_deg: float  # angle between the spin axis and the vertical camera axis
    accepted: int
    rejected: int
    axis: np.ndarray | None = None  # unit spin axis (camera frame), None without enough rotation

    @property
    def rejected_share(self) -> float:
        return self.rejected / max(1, self.accepted + self.rejected)


@dataclass
class TurnCounter:
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))  # the camera looks down
    max_step_deg: float = 40.0
    max_tilt_deg: float = 30.0

    def __post_init__(self):
        self.up = np.asarray(self.up, dtype=float) / np.linalg.norm(self.up)
        self.reset()

    def reset(self):
        self.detections: list[Detection] = []
        self.reference: Rotation | None = None
        self.rejected_in_row = 0
        self.angle_camera_axis = 0.0
        self.accepted = 0
        self.rejected = 0

    def add(self, time: float, position: np.ndarray, rotation: Rotation) -> Detection:
        position = np.asarray(position, dtype=float)
        if self.reference is None:
            self.reference = rotation
            detection = Detection(time, position, rotation, True, np.zeros(3))
            self.detections.append(detection)
            return detection
        candidates = rotation * CUBE_SYMMETRIES
        deltas = candidates * self.reference.inv()
        best = int(np.argmin(deltas.magnitude()))
        step = deltas[best].as_rotvec()
        about_up = float(step @ self.up)
        off_axis = float(np.linalg.norm(step - about_up * self.up))
        plausible = abs(about_up) <= np.radians(self.max_step_deg) and off_axis <= np.radians(self.max_tilt_deg)
        if not plausible:
            self.rejected += 1
            self.rejected_in_row += 1
            if self.rejected_in_row < REANCHOR_AFTER:
                detection = Detection(time, position, candidates[best], False, step)
                self.detections.append(detection)
                return detection
            step, about_up = np.zeros(3), 0.0  # a real change of pose that could not be followed: re-anchor
        else:
            self.accepted += 1
        self.rejected_in_row = 0
        self.reference = candidates[best]
        self.angle_camera_axis += about_up
        detection = Detection(time, position, candidates[best], True, step)
        self.detections.append(detection)
        return detection

    @property
    def turns_camera_axis(self) -> float:
        return self.angle_camera_axis / (2 * np.pi)

    def result(self) -> TurnResult:
        steps = np.array([d.step for d in self.detections if d.accepted]).reshape(-1, 3)
        total = steps.sum(axis=0)
        angle, tilt, axis = self.angle_camera_axis, 0.0, None
        if np.linalg.norm(total) > np.radians(90):  # enough net rotation to define a spin axis
            axis = total / np.linalg.norm(total)
            axis = axis if axis @ self.up >= 0 else -axis
            angle = float((steps @ axis).sum())
            tilt = float(np.degrees(np.arccos(np.clip(axis @ self.up, -1.0, 1.0))))
        return TurnResult(angle / (2 * np.pi), self.turns_camera_axis, tilt, self.accepted, self.rejected, axis)

    def progress(self, axis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(times, cumulative turns) at the accepted detections, about axis (default: the vertical camera axis)."""
        axis = self.up if axis is None else axis
        accepted = [d for d in self.detections if d.accepted]
        times = np.array([d.time for d in accepted])
        turns = np.cumsum([d.step @ axis for d in accepted]) / (2 * np.pi) if accepted else np.zeros(0)
        return times, turns

    def positions_since(self, since: float) -> list[tuple[float, np.ndarray]]:
        return [(d.time, d.position) for d in self.detections if d.time >= since]


@dataclass
class DropMonitor:
    """Decides whether the cube fell, from the median position of the last second, so that single wrong detections
    do not count. The cube must stay more than drop_dist from where it rested for drop_confirm seconds."""

    rest_position: np.ndarray
    drop_dist: float = 0.06
    drop_confirm: float = 1.0
    window: float = 1.0
    min_detections: int = 3
    far_since: float | None = None
    distance: float = 0.0  # m, of the median position from the rest position

    def update(self, now: float, counter: TurnCounter) -> bool:
        recent = [p for t, p in counter.positions_since(now - self.window)]
        if len(recent) >= self.min_detections:
            self.distance = float(np.linalg.norm(np.median(np.array(recent), axis=0) - self.rest_position))
            if self.distance > self.drop_dist:
                self.far_since = self.far_since if self.far_since is not None else now
            else:
                self.far_since = None
        return self.far_since is not None and now - self.far_since >= self.drop_confirm


def resting_position(samples: list[np.ndarray], max_spread: float = 0.005) -> np.ndarray | None:
    """The median of the samples if the cube lies still (all within max_spread of the median), else None."""
    if len(samples) < 5:
        return None
    samples = np.array(samples)
    median = np.median(samples, axis=0)
    if np.percentile(np.linalg.norm(samples - median, axis=1), 90) > max_spread:
        return None
    return median
