"""The live view and the video frame: the camera image with the detected cube drawn into it and a status panel."""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

GREEN, YELLOW, ORANGE, RED, WHITE, GRAY = (80, 220, 80), (0, 230, 255), (0, 150, 255), (60, 60, 255), (255, 255, 255), (170, 170, 170)
CUBE_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


@dataclass
class ViewState:
    title: str = "real policy evaluation"
    phase: str = ""
    turns: float | None = None  # about the camera axis, shown during a run
    rejected_share: float = 0.0
    detections_per_s: float = 0.0
    pose_age: float | None = None  # s since the last detection
    position: np.ndarray | None = None  # m, camera frame
    rotation: Rotation | None = None
    trail: list = field(default_factory=list)  # recent positions, camera frame
    rest_position: np.ndarray | None = None
    drop_dist: float = 0.06
    temps: np.ndarray | None = None
    joint_names: list = field(default_factory=list)
    start_temp: float = 50.0
    abort_temp: float = 58.0


def _project(points, K, dist):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if np.any(points[:, 2] <= 0.01):
        return None
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, dist)
    return pixels.reshape(-1, 2)


def _cube_corners(position, rotation, edge):
    signs = np.array([[x, y, z] for z in (1, -1) for y in (1, -1) for x in (1, -1)], dtype=float)
    return rotation.apply(signs * edge / 2) + position


def _text(img, text, xy, color, scale=0.6, thickness=1):
    # a shadow with the same thickness: a thicker outline changes the glyph widths in OpenCV 5
    x, y = xy
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (2, 2)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class VideoRecorder:
    """Writes frames in real time: frames are repeated when they arrive slower than fps."""

    def __init__(self, path, fps=20.0):
        self.path, self.fps = str(path), fps
        self.writer, self.t0, self.frames = None, None, 0

    def write(self, frame, now):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.t0 = now
        due = int((now - self.t0) * self.fps) + 1
        while self.frames < due:
            self.writer.write(frame)
            self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def temperature_color(temp, start_temp, abort_temp):
    if temp >= abort_temp:
        return RED
    if temp >= start_temp:
        return ORANGE
    return GREEN


def render(image: np.ndarray, state: ViewState, K: np.ndarray | None, dist: np.ndarray | None,
           cube_edge: float) -> np.ndarray:
    frame = image.copy()
    h, w = frame.shape[:2]
    dist = np.zeros(5) if dist is None or len(dist) == 0 else np.asarray(dist, dtype=np.float64)

    if K is not None:
        # where the cube rested before the run, with the drop radius
        if state.rest_position is not None:
            center = _project(state.rest_position, K, dist)
            if center is not None:
                c = tuple(int(v) for v in center[0])
                radius = int(K[0, 0] * state.drop_dist / state.rest_position[2])
                cv2.circle(frame, c, radius, GRAY, 1, cv2.LINE_AA)
                cv2.drawMarker(frame, c, GRAY, cv2.MARKER_CROSS, 14, 1)
        # the path of the cube center
        if len(state.trail) >= 2:
            pts = _project(state.trail, K, dist)
            if pts is not None:
                cv2.polylines(frame, [pts.astype(np.int32)], False, YELLOW, 1, cv2.LINE_AA)
        # the detected cube: green while fresh, orange when the last detection is older than 0.25 s
        if state.position is not None and state.rotation is not None and state.pose_age is not None and state.pose_age < 1.0:
            color = GREEN if state.pose_age <= 0.25 else ORANGE
            corners = _project(_cube_corners(state.position, state.rotation, cube_edge), K, dist)
            if corners is not None:
                for a, b in CUBE_EDGES:
                    cv2.line(frame, tuple(corners[a].astype(int)), tuple(corners[b].astype(int)), color, 2, cv2.LINE_AA)
                axes = _project([state.position] + [state.position + state.rotation.apply(v * cube_edge * 0.8)
                                                    for v in np.eye(3)], K, dist)
                if axes is not None:
                    for i, c in enumerate([(0, 0, 255), (0, 200, 0), (255, 80, 0)]):  # x red, y green, z blue
                        cv2.line(frame, tuple(axes[0].astype(int)), tuple(axes[i + 1].astype(int)), c, 2, cv2.LINE_AA)

    # status panel
    lines = [(state.title, WHITE, 0.65), (state.phase, YELLOW, 0.6)]
    if state.turns is not None:
        lines.append((f"turns about the camera axis {state.turns:+.2f}", YELLOW, 0.6))
    if state.pose_age is None or state.pose_age > 1.0:
        lines.append(("cube NOT DETECTED", RED, 0.55))
    else:
        tracked = "tracked" if state.pose_age <= 0.25 else f"last seen {state.pose_age:.1f} s ago"
        lines.append((f"cube {tracked}, {state.detections_per_s:.0f} poses/s, rejected {state.rejected_share:.0%}",
                      GREEN if state.pose_age <= 0.25 else ORANGE, 0.55))
    if state.temps is not None and len(state.temps):
        order = np.argsort(state.temps)[::-1][:3]
        hottest = "   ".join(f"{state.joint_names[i] if i < len(state.joint_names) else i} {state.temps[i]:.0f}C"
                               for i in order)
        color = temperature_color(float(state.temps[order[0]]), state.start_temp, state.abort_temp)
        lines.append((f"hottest motors: {hottest}", color, 0.55))
    else:
        lines.append(("no motor temperatures", GRAY, 0.55))
    panel_h = 12 + 26 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(w, 620), panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    for i, (text, color, scale) in enumerate(lines):
        _text(frame, text, (10, 26 + 26 * i), color, scale, 2 if i == 0 else 1)
    return frame
