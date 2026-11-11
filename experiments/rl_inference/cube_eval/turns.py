"""Count the cube's turns and detect a drop from detected cube poses in the camera frame. No ROS, no camera.

The counter predicts the cube's orientation from the last accepted detection and the recent spin rate. A detection
further than ``max_residual_deg`` from the prediction is a wrong detection and is ignored; if the next detection
shows that the last accepted one was the wrong one, that one is taken back. Time without an accepted detection for
longer than ``max_gap`` seconds is added to ``gap_s``; the next detection is then taken whatever the prediction says.

The detector tells the faces of the lettered cube apart (on the hand, 3 of 305 detections named them differently
from the one before), so by default an orientation is taken as it is, and the count is right as long as the cube's
motion differs by less than 180 degrees from the prediction between two detections. For a cube whose faces look
alike, ``symmetric=True`` compares each detection in all 24 orientations in which a cube looks the same and takes
the closest; the limit is then 45 degrees, max_residual_deg must be well below that (25), and the rotation during
a gap is not counted.

Times are the caller's clock. eval_policies.py passes the active run time, which stands still during a pause, so
speeds are per second of policy time.
"""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

CUBE_SYMMETRIES = Rotation.create_group("O")  # the 24 rotations that map a cube onto itself


class TurnCounter:
    """Accumulates the cube's rotation about ``up`` (unit vector in the camera frame, default: toward the camera)."""

    def __init__(self, up=(0.0, 0.0, -1.0), symmetric=False, max_residual_deg=None, max_gap=0.6, velocity_gain=0.3):
        self.up = np.asarray(up, dtype=float) / np.linalg.norm(up)
        self.symmetric = symmetric
        self.max_residual = np.radians(max_residual_deg if max_residual_deg is not None else 25.0 if symmetric else 60.0)
        self.max_gap = max_gap
        self.velocity_gain = velocity_gain  # of the newest step in the running mean of the spin rate
        self.reference = None      # the last accepted orientation
        self.anchor = None         # the orientation headings are measured against
        self._tilted_before = False
        self.reference_time = None
        self.velocity = np.zeros(3)  # rad/s, rotation vector rate in the camera frame
        self.accepted = self.rejected = 0
        self.angle = 0.0        # rad about up, signed
        self.tilt_angle = 0.0   # rad about the other axes, added up
        self.gap_s = 0.0        # time without an accepted detection, in stretches longer than max_gap
        self.history = []       # (time, angle) at every accepted detection
        self._previous = None   # the state before the last accepted detection, to take that one back

    def rebase(self):
        """Forget the reference orientation: the next detection starts a new segment (after a pause, when a person
        may have turned the cube)."""
        self.reference = None
        self.velocity = np.zeros(3)
        self._previous = None

    def add(self, time, rotation):
        """Add a detection (scipy Rotation, cube in camera frame). Returns True if it was accepted."""
        after_gap = self.reference is not None and time - self.reference_time > self.max_gap
        if after_gap:
            self.gap_s += time - self.reference_time
            if self.symmetric:  # which of the 24 orientations it is can no longer be told
                self.rebase()
        if self.reference is None:
            self.reference, self.reference_time = rotation, time
            self.anchor = rotation
            self.history.append((time, self.angle))
            return True
        step = self._match(time, rotation, force=after_gap)
        if step is None and self._previous is not None:
            # it does not follow the last accepted detection; if it follows the one before, the last one was wrong
            current = self._state()
            self._restore(self._previous)
            step = self._match(time, rotation)
            if step is None:
                self._restore(current)
            else:
                self.rejected += 1  # the detection that was taken back
        if step is None:
            self.rejected += 1
            return False
        self._previous = self._state()
        matched, step = step
        dt = time - self.reference_time
        # the change of the cube's heading: the difference of two headings against a fixed orientation, not the
        # step's own rotation, so that the noise of the single detections does not add up over a run
        # (a heading is ill-defined for an orientation tilted far against the anchor: then the step's own rotation)
        if max(self._swing(matched), self._swing(self.reference)) < np.radians(60):
            about_up = self._heading(matched) - self._heading(self.reference)
            about_up = float((about_up + np.pi) % (2 * np.pi) - np.pi)
        else:
            about_up = float(step @ self.up)
        self.angle += about_up
        self.tilt_angle += float(np.linalg.norm(step - about_up * self.up))
        if dt > 1e-3:
            self.velocity = (1 - self.velocity_gain) * self.velocity + self.velocity_gain * step / dt
        self.reference, self.reference_time = matched, time
        # the heading against the anchor is ill-defined near 180 deg of tilt: take a new anchor when the cube has
        # tilted far, but only once two detections in a row say so (one may be a wrong one)
        tilted = self._swing(matched) > np.radians(60)
        if tilted and self._tilted_before:
            self.anchor = matched
        self._tilted_before = tilted and self.anchor is not matched
        self.accepted += 1
        self.history.append((time, self.angle))
        return True

    def _match(self, time, rotation, force=False):
        """(the orientation, the step to it from the reference as a rotation vector), or None for a wrong detection."""
        predicted = Rotation.from_rotvec(self.velocity * (time - self.reference_time)) * self.reference
        if self.symmetric:
            candidates = rotation * CUBE_SYMMETRIES
            residuals = (candidates * predicted.inv()).magnitude()
            rotation = candidates[int(np.argmin(residuals))]
        if not force and (rotation * predicted.inv()).magnitude() > self.max_residual:
            return None
        step = (rotation * self.reference.inv()).as_rotvec()
        if np.linalg.norm(step) > np.pi * 0.99:  # a half turn has no direction
            return None
        return rotation, step

    def _relative(self, rotation):
        q = (rotation * self.anchor.inv()).as_quat()
        return float(q[:3] @ self.up), float(q[3])

    def _heading(self, rotation):
        """The rotation about up that takes the anchor's orientation closest to this one (rad)."""
        along, w = self._relative(rotation)
        return 2.0 * np.arctan2(along, w)

    def _swing(self, rotation):
        """How far this orientation is tilted against the anchor's, apart from the heading (rad)."""
        along, w = self._relative(rotation)
        return 2.0 * np.arccos(np.clip(np.hypot(along, w), 0.0, 1.0))

    def _state(self):
        return (self.reference, self.reference_time, self.velocity.copy(), self.angle, self.tilt_angle,
                self.accepted, len(self.history), self.anchor)

    def _restore(self, state):
        (self.reference, self.reference_time, self.velocity, self.angle, self.tilt_angle, self.accepted, n,
         self.anchor) = state
        self.velocity = self.velocity.copy()
        del self.history[n:]

    @property
    def turns(self):
        """Net turns, positive = counterclockwise seen from the camera side of ``up``."""
        return self.angle / (2 * np.pi)

    @property
    def rejected_share(self):
        return self.rejected / max(1, self.accepted + self.rejected)

    def angle_at(self, time):
        """The accumulated angle at a time (the last accepted detection before it)."""
        if not self.history or time < self.history[0][0]:
            return 0.0
        times = [t for t, _ in self.history]
        return self.history[int(np.searchsorted(times, time, side="right")) - 1][1]

    def speed(self, now, window=5.0):
        """Mean angular velocity about up over the last ``window`` seconds (rad/s); over the time since the start
        if that is shorter."""
        if not self.history:
            return 0.0
        span = min(window, now - self.history[0][0])
        if span <= 0.2:
            return 0.0
        return (self.angle - self.angle_at(now - span)) / span


def lateral_offset(position, home):
    """Distance of the cube from its home position across the camera's view (m), measured at the home depth.

    From above, the detector's depth is its least certain value, so only the direction in which the cube is seen
    counts: the position is scaled to the home depth before the comparison."""
    position, home = np.asarray(position, dtype=float), np.asarray(home, dtype=float)
    if position[2] <= 1e-3:
        return float("inf")
    return float(np.linalg.norm(position[:2] * home[2] / position[2] - home[:2]))


class RestDetector:
    """Decides whether the cube lies still: the positions of the last ``window`` seconds are within ``spread``."""

    def __init__(self, window=1.5, spread=0.006, min_detections=5):
        self.window, self.spread, self.min_detections = window, spread, min_detections
        self.samples = deque()

    def add(self, time, position):
        self.samples.append((time, np.asarray(position, dtype=float)))

    def resting(self, now):
        """The median position if the cube lies still and was seen until now, else None."""
        while self.samples and self.samples[0][0] < now - self.window:
            self.samples.popleft()
        if len(self.samples) < self.min_detections or now - self.samples[-1][0] > 0.5 * self.window:
            return None
        if self.samples[-1][0] - self.samples[0][0] < 0.6 * self.window:
            return None  # the window is not covered yet
        positions = np.array([p for _, p in self.samples])
        median = np.median(positions, axis=0)
        offsets = [lateral_offset(p, median) for p in positions]
        return median if np.percentile(offsets, 90) <= self.spread else None


class DropMonitor:
    """The cube fell when it stays more than ``drop_dist`` (m, across the view) from its home position for
    ``confirm`` seconds, or is not seen for ``lost_timeout`` seconds. Single wrong detections do not count: the
    median of the last ``window`` seconds is compared."""

    def __init__(self, home, drop_dist=0.06, confirm=0.5, lost_timeout=3.0, window=0.5):
        self.home = np.asarray(home, dtype=float)
        self.drop_dist, self.confirm, self.lost_timeout, self.window = drop_dist, confirm, lost_timeout, window
        self.samples = deque()
        self.far_since = None
        self.last_seen = None
        self.offset = 0.0       # m, of the recent median from home
        self.max_offset = 0.0
        self.offsets = []       # every detection's offset, for the mean drift

    def start(self, now):
        """(Re)start watching: the cube counts as seen now."""
        self.samples.clear()
        self.far_since, self.last_seen = None, now

    def add(self, now, position):
        self.samples.append((now, np.asarray(position, dtype=float)))
        self.last_seen = now
        self.offsets.append(lateral_offset(position, self.home))

    def check(self, now):
        """"" while the cube is in the hand, else the reason: "dropped" or "lost"."""
        while self.samples and self.samples[0][0] < now - self.window:
            self.samples.popleft()
        if len(self.samples) >= 2:
            median = np.median(np.array([p for _, p in self.samples]), axis=0)
            self.offset = lateral_offset(median, self.home)
            self.max_offset = max(self.max_offset, self.offset)
            if self.offset > self.drop_dist:
                self.far_since = now if self.far_since is None else self.far_since
            else:
                self.far_since = None
        if self.far_since is not None and now - self.far_since >= self.confirm:
            return "dropped"
        if self.last_seen is not None and now - self.last_seen >= self.lost_timeout:
            return "lost"
        return ""
