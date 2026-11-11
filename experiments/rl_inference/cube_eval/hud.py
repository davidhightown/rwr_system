"""Draw the evaluation's view: the camera image with the detected cube, the path of its centre and the run's figures.
No ROS, no camera: render() takes an image and plain values."""

import cv2
import numpy as np

GREEN, YELLOW, ORANGE, RED, WHITE, GRAY = (80, 220, 80), (0, 230, 255), (0, 150, 255), (60, 60, 255), (255, 255, 255), (170, 170, 170)
COLORS = dict(green=GREEN, yellow=YELLOW, orange=ORANGE, red=RED, white=WHITE, gray=GRAY)
CUBE_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def project(points, K, dist=None):
    """Pixels of camera-frame points (n, 3); None if a point is behind the camera."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if np.any(points[:, 2] <= 0.01):
        return None
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), np.asarray(K, dtype=np.float64), dist)
    return pixels.reshape(-1, 2)


def text(img, string, xy, color, scale=0.55, thickness=1):
    x, y = xy  # a shadow of the same thickness: a thicker outline has other glyph widths in OpenCV 5
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (2, 2)):
        cv2.putText(img, string, (x + dx, y + dy), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, string, xy, FONT, scale, color, thickness, cv2.LINE_AA)


def status_lines(s):
    """The panel's (text, colour) lines from the status dict of eval_policies.py."""
    lines = [(s.get("model", ""), WHITE), (s.get("phase", ""), COLORS.get(s.get("phase_color", "yellow"), YELLOW))]
    if s.get("duration"):
        lines.append((f"time {s.get('elapsed', 0.0):5.1f} / {s['duration']:.0f} s    drops {s.get('drops', 0)}", WHITE))
        lines.append((f"turns {s.get('turns', 0.0):+6.2f}    speed (5 s) {s.get('speed', 0.0):+5.2f} rad/s    "
                      f"goal {s.get('goal', 0.0):+.1f}", YELLOW))
    temps = s.get("temps")
    if temps is not None and len(temps) and np.any(np.isfinite(temps)):
        temps = np.asarray(temps, dtype=float)
        hot = int(np.nanargmax(temps))
        names = s.get("joint_names") or []
        name = names[hot] if hot < len(names) else str(hot)
        limit = s.get("temp_limit", 60.0)
        color = RED if temps[hot] >= limit else ORANGE if temps[hot] >= limit - 8 else GREEN
        lines.append((f"motors mean {np.nanmean(temps):.0f} C    hottest {name} {temps[hot]:.0f} C (limit {limit:.0f})",
                      color))
    else:
        lines.append(("no motor temperatures", GRAY))
    return lines


def render(image, status, detection, trail, K, dist=None, cube_edge=0.045, detection_age=None, detection_hz=0.0):
    """image: BGR. detection: (R, t, corner pixels) of the last detection or None. trail: [(x, y)] pixels of the cube
    centre, oldest first. Returns a new image."""
    frame = image.copy()
    h, w = frame.shape[:2]

    home = status.get("home")
    if home is not None and K is not None:
        centre = project(home, K, dist)
        if centre is not None:
            c = (int(centre[0][0]), int(centre[0][1]))
            cv2.circle(frame, c, int(K[0][0] * status.get("drop_dist", 0.06) / home[2]), GRAY, 1, cv2.LINE_AA)
            cv2.drawMarker(frame, c, GRAY, cv2.MARKER_CROSS, 12, 1)

    if len(trail) >= 2:  # the path of the cube's centre, fading toward the past
        pts = np.asarray(trail, dtype=np.int32)
        n = len(pts) - 1
        for i in range(n):
            fade = (i + 1) / n
            cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), (0, int(120 + 110 * fade), int(255 * fade)),
                     1 if fade < 0.5 else 2, cv2.LINE_AA)

    fresh = detection is not None and detection_age is not None and detection_age < 1.0
    if fresh and K is not None:
        R, t, _ = detection
        color = GREEN if detection_age < 0.5 else ORANGE
        signs = np.array([[x, y, z] for z in (1, -1) for y in (1, -1) for x in (1, -1)], dtype=float)
        corners = project(signs * cube_edge / 2 @ np.asarray(R).T + t, K, dist)
        if corners is not None:
            for a, b in CUBE_EDGES:
                cv2.line(frame, tuple(corners[a].astype(int)), tuple(corners[b].astype(int)), color, 2, cv2.LINE_AA)
        axes = project([t] + [t + np.asarray(R)[:, i] * cube_edge * 0.8 for i in range(3)], K, dist)
        if axes is not None:
            for i, c in enumerate([(0, 0, 255), (0, 200, 0), (255, 80, 0)]):  # x red, y green, z blue
                cv2.line(frame, tuple(axes[0].astype(int)), tuple(axes[i + 1].astype(int)), c, 2, cv2.LINE_AA)

    lines = status_lines(status)
    if fresh:
        lines.append((f"cube tracked, {detection_hz:.1f} poses/s", GREEN if detection_age < 0.5 else ORANGE))
    else:
        lines.append(("cube NOT DETECTED", RED))
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (min(w, 560), 14 + 24 * len(lines)), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.5, frame, 0.5, 0)
    for i, (string, color) in enumerate(lines):
        text(frame, string, (10, 26 + 24 * i), color, 0.62 if i == 0 else 0.55, 2 if i == 0 else 1)

    banner = status.get("banner")
    if banner:
        scale = 0.9
        (tw, th), _ = cv2.getTextSize(banner, FONT, scale, 2)
        scale = min(scale, scale * (w - 40) / max(tw, 1))
        (tw, th), _ = cv2.getTextSize(banner, FONT, scale, 2)
        x, y = (w - tw) // 2, h - 40
        box = frame.copy()
        cv2.rectangle(box, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
        frame = cv2.addWeighted(box, 0.6, frame, 0.4, 0)
        text(frame, banner, (x, y), COLORS.get(status.get("banner_color", "yellow"), YELLOW), scale, 2)
    return frame
