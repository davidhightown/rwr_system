"""The full-screen demo view: header with the model and the runs, the camera image (or the digital twin and the joint
state without a camera), a chart over the run, the timer, figures and motor temperatures. No ROS, no camera:
Dashboard.render() takes the status dict of eval_policies.py (with its "demo" dict) and the images, and returns a BGR
image. It keeps the plotted history itself and is drawn with OpenCV only (the static parts once per layout)."""

import math
import re
import sys
import time
import traceback
from functools import lru_cache

import cv2
import numpy as np

SANS, NUM = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA


def rgb(r, g, b):
    return (b, g, r)


BG, HEADER, CARD, INNER, EDGE = rgb(13, 16, 22), rgb(19, 23, 31), rgb(25, 30, 40), rgb(37, 44, 57), rgb(48, 56, 72)
TXT, SUB, MUTED = rgb(238, 241, 246), rgb(152, 162, 178), rgb(98, 108, 124)
GREEN, YELLOW, ORANGE, RED = rgb(70, 214, 120), rgb(255, 208, 70), rgb(255, 144, 48), rgb(245, 72, 72)
BLUE, PURPLE, TEAL, AMBER, WHITE = rgb(72, 160, 255), rgb(170, 124, 255), rgb(46, 210, 196), rgb(255, 186, 56), TXT
COLORS = dict(green=GREEN, yellow=YELLOW, orange=ORANGE, red=RED, white=WHITE)
MODE_COLORS = dict(proprio=PURPLE, proprio_camera=PURPLE, extero=BLUE, axes=TEAL, goals=AMBER)  # proprio_camera:
# the proprio policy with the camera plugged in (the camera view and the turns, as extero)
STATE_COLORS = dict(running=GREEN, cooling=ORANGE, waiting=YELLOW, countdown=YELLOW, paused=WHITE, stopped=RED,
                    done=WHITE, starting=SUB)
GROUPS = ("thumb", "index", "middle", "ring", "pinky", "wrist")
PARTS = ("cmc", "abd", "mcp", "pip", "dip")
G = 16  # gutter
MAIN = (16, 100, 976, 640)
CHART = (16, 656, 976, 884)
STATE_CARD, MOTORS, FOOTER = (992, 100, 1584, 232), (992, 504, 1584, 796), (992, 812, 1584, 884)
TILES = [(992, 248, 1280, 360), (1296, 248, 1584, 360), (992, 376, 1280, 488), (1296, 376, 1584, 488)]
HISTORY_S = 90.0  # at least the whole run (duration 60 s) stays on the chart


def num(x):
    """A finite float or None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def nums(xs, n=None):
    """A float array (NaN for missing), or None."""
    if xs is None:
        return None
    try:
        out = np.array([np.nan if num(v) is None else num(v) for v in xs], dtype=float)
    except TypeError:
        return None
    if n is not None and len(out) != n:
        out = np.resize(out, n) if len(out) else np.full(n, np.nan)
    return out


def seq(xs):
    return list(xs) if isinstance(xs, (list, tuple, np.ndarray)) else []


def count(x, hi=8):
    return int(np.clip(num(x) or 0, 0, hi))


def seconds(x):
    v = num(x)
    return v if v is not None and 1.0 <= v <= 3600 else 60.0


@lru_cache(maxsize=4096)
def tsize(s, scale, thick=1, font=SANS):
    (w, h), _ = cv2.getTextSize(s, font, scale, thick)
    return w, h


def put(img, s, x, y, scale, color, thick=1, font=SANS, anchor="l", max_w=None):
    """Text with its baseline at y; anchor l/c/r. Shrinks to max_w. Returns the drawn width."""
    s = str(s)
    w, _ = tsize(s, scale, thick, font)
    if max_w and w > max_w:
        scale *= max_w / w
        w, _ = tsize(s, scale, thick, font)
    x = int(x - w / 2) if anchor == "c" else int(x - w) if anchor == "r" else int(x)
    cv2.putText(img, s, (x, int(y)), font, scale, color, thick, AA)
    return w


def shadow(img, s, x, y, scale, color, thick=2, font=SANS, anchor="c"):
    w, _ = tsize(s, scale, thick, font)
    x0 = x - w / 2 if anchor == "c" else x
    for d in (2, 3):
        cv2.putText(img, s, (int(x0 + d), int(y + d)), font, scale, (0, 0, 0), thick, AA)
    cv2.putText(img, s, (int(x0), int(y)), font, scale, color, thick, AA)


def rrect(img, x0, y0, x1, y1, color, r=10, thick=-1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if thick < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
    else:
        for p, q in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)), ((x0, y0 + r), (x0, y1 - r)),
                     ((x1, y0 + r), (x1, y1 - r))):
            cv2.line(img, p, q, color, thick, AA)
        for (cx, cy), a in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x1 - r, y1 - r), 0),
                            ((x0 + r, y1 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thick, AA)


def card(img, rect, title=None, color=CARD):
    rrect(img, *rect, color, 12)
    if title:
        put(img, title, rect[0] + 18, rect[1] + 28, 0.5, SUB, 1)


def blend(img, rect, color, alpha):
    """Darkens/tints a rectangle in place (alpha = weight of the colour)."""
    x0, y0, x1, y1 = rect
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0, dst=roi)


def dashed(img, p, q, color, thick=1, dash=8, gap=6):
    p, q = np.array(p, float), np.array(q, float)
    length = np.linalg.norm(q - p)
    if length < 1:
        return
    d = (q - p) / length
    for s in np.arange(0, length, dash + gap):
        a, b = p + d * s, p + d * min(s + dash, length)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, thick, AA)


def temp_color(t, start, limit):
    """Green up to the start temperature, to orange, to red at the limit."""
    if t is None:
        return MUTED
    f = float(np.clip((t - start) / max(limit - start, 1.0), 0, 1))
    a, b, k = (GREEN, ORANGE, f / 0.6) if f < 0.6 else (ORANGE, RED, (f - 0.6) / 0.4)
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def nice_step(span, ticks=4):
    raw = span / ticks
    mag = 10 ** math.floor(math.log10(max(raw, 1e-9)))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def joint_order(names):
    """Joint indices grouped by finger (GROUPS order) and part (PARTS order), with (group, part) labels."""
    def key(i):
        group, _, part = names[i].partition("_")
        return (GROUPS.index(group) if group in GROUPS else len(GROUPS),
                PARTS.index(part) if part in PARTS else len(PARTS), i)
    order = sorted(range(len(names)), key=key)
    return order, [tuple(names[i].partition("_")[::2]) for i in order]


class Dashboard:
    def __init__(self, size=(1600, 900)):
        self.size = tuple(size)
        self.static_key, self.static = None, None
        self.trail_id, self.hist = object(), {}
        self.twin_cache = (None, None)
        self.last_error = None

    # history ---------------------------------------------------------------------------------------------------
    def _record(self, s, d):
        if s.get("trail_id") != self.trail_id:
            self.trail_id = s.get("trail_id")
            self.hist = {k: [] for k in ("t", "turns", "err", "goals", "hot", "track")}
        t = num(s.get("elapsed"))
        if not s.get("running") or t is None:
            return
        h = self.hist
        if h["t"] and t <= h["t"][-1]:
            if t < h["t"][-1] - 1.0:  # time went back without a new trail_id: a new run
                for v in h.values():
                    v.clear()
            else:
                return
        temps = nums(s.get("temps"))
        hot = float(np.nanmax(temps)) if temps is not None and np.isfinite(temps).any() else np.nan
        for k, v in (("t", t), ("turns", num(s.get("turns"))), ("err", num(s.get("goal_error_deg"))),
                     ("goals", num(s.get("goals"))), ("hot", hot), ("track", num(d.get("track_err_deg")))):
            h[k].append(np.nan if v is None else v)
        while h["t"] and h["t"][0] < t - HISTORY_S:
            for v in h.values():
                del v[0]

    # static layer ----------------------------------------------------------------------------------------------
    def _layout(self, mode, has_cam, has_twin):
        if has_cam:
            return None, None
        x0, y0, x1, y1 = MAIN
        if has_twin:
            tw = int(round(300 * (y1 - y0) / 380))
            return (x0, y0, x0 + tw, y1), (x0 + tw + G, y0, x1, y1)
        return None, MAIN

    def _build_static(self, s, d, mode, has_cam, has_twin, names):
        W, H = self.size
        img = np.full((H, W, 3), BG, np.uint8)
        accent = MODE_COLORS.get(mode, BLUE)
        cv2.rectangle(img, (0, 0), (W, 84), HEADER, -1)
        cv2.line(img, (0, 84), (W, 84), EDGE, 1)
        cv2.rectangle(img, (0, 0), (6, 84), accent, -1)
        put(img, d.get("title") or "ORCA Hand", 26, 38, 0.82, TXT, 2, max_w=600)
        label = str(d.get("mode_label") or mode or "").upper()
        if label:
            w, _ = tsize(label, 0.48, 1)
            rrect(img, 26, 50, 26 + w + 24, 74, accent, 12)
            put(img, label, 38, 67, 0.48, BG, 1)
        put(img, str(d.get("model") or "").upper(), 840, 44, 1.3, TXT, 2, NUM, "c", max_w=380)
        put(img, d.get("task") or "", 840, 72, 0.56, SUB, 1, anchor="c", max_w=380)

        twin_rect, joints_rect = self._layout(mode, has_cam, has_twin)
        if twin_rect:
            card(img, twin_rect, color=INNER)
        if joints_rect:
            self._joints_static(img, joints_rect, names, d)

        x0, y0, x1, y1 = CHART
        card(img, CHART, {"goals": "ANGLE TO THE GOAL ORIENTATION", "proprio": "MOTORS AND TRACKING"}.get(
            mode, "CUBE TURNS"))
        card(img, STATE_CARD)
        tiles = {"goals": ("GOALS REACHED", "ANGLE TO GOAL", "SECONDS PER GOAL", "HOTTEST MOTOR"),
                 "proprio": ("CONTROL LOOP", "TRACKING ERROR", "PEAK CURRENT", "HOTTEST MOTOR")}.get(
            mode, ("CUBE TURNS", "SPEED (LAST 5 S)", "CUBE TRACKING", "HOTTEST MOTOR"))
        for rect, t in zip(TILES, tiles):
            card(img, rect, t)
        card(img, MOTORS, "MOTOR TEMPERATURE")
        self._motors_static(img, names, d, s)
        card(img, FOOTER)
        self._footer_static(img, d)
        return img

    def _joints_static(self, img, rect, names, d):
        card(img, rect, "JOINT STATE")
        x0, y0, x1, y1 = rect
        w = put(img, "commanded", x1 - 18, y0 + 28, 0.45, SUB, 1, anchor="r")
        cv2.line(img, (x1 - 30 - w, y0 + 14), (x1 - 30 - w, y0 + 30), YELLOW, 2, AA)
        w2 = put(img, "measured", x1 - 50 - w, y0 + 28, 0.45, SUB, 1, anchor="r")
        rrect(img, x1 - 84 - w - w2, y0 + 17, x1 - 60 - w - w2, y0 + 27, PURPLE, 3)
        order, labels = joint_order(names)
        top, gap = y0 + 48, 9
        ngroups = len({g for g, _ in labels})
        row = (y1 - 14 - top - gap * max(ngroups - 1, 0)) / max(len(order), 1)
        self.rows, y, prev = [], top, None
        bx0, bx1 = x0 + 150, x1 - 76
        for j, (g, p) in zip(order, labels):
            if prev is not None and g != prev:
                y += gap
            yc = int(y + row / 2)
            if g != prev:
                put(img, g.upper(), x0 + 18, yc + 6, 0.5, TXT, 1)
            put(img, p or "", x0 + 92, yc + 5, 0.45, SUB, 1)
            h = int(min(row * 0.5, 12))
            rrect(img, bx0, yc - h // 2, bx1, yc + h // 2, INNER, 3)
            self.rows.append((j, yc, h))
            y += row
            prev = g
        self.bar_x = (bx0, bx1)

    def _motors_static(self, img, names, d, s):
        x0, y0, x1, y1 = MOTORS
        order, labels = joint_order(names)
        n, ngroups = len(order), len({g for g, _ in labels})
        px0, px1, gap = x0 + 18, x1 - 52, 10
        w = (px1 - px0 - gap * max(ngroups - 1, 0)) / max(n, 1)
        self.mbars, x, prev = [], px0, None
        groups = {}
        for j, (g, p) in zip(order, labels):
            if prev is not None and g != prev:
                x += gap
            self.mbars.append((j, int(x + 3), int(x + w - 3)))
            groups.setdefault(g, [x, x + w])[1] = x + w
            put(img, (p or g)[:3], x + w / 2, y0 + 180, 0.38, SUB, 1, anchor="c")
            x += w
            prev = g
        for g, (a, b) in groups.items():
            cv2.line(img, (int(a + 3), y0 + 188), (int(b - 3), y0 + 188), EDGE, 1, AA)
            put(img, g.upper(), (a + b) / 2, y0 + 204, 0.4, TXT, 1, anchor="c", max_w=b - a + gap)
        put(img, "MOTOR CURRENT", x0 + 18, y0 + 234, 0.5, SUB, 1)
        self.temp_plot = (px0, y0 + 50, px1, y0 + 160)
        self.cur_plot = (y0 + 246, y1 - 14)

    def _footer_static(self, img, d):
        x0, y0, x1, y1 = FOOTER
        x, y = x0 + 14, y0 + 12
        for item in seq(d.get("params")):
            try:
                k, v = str(item[0]), str(item[1])
            except (TypeError, IndexError, KeyError):
                continue
            wk, _ = tsize(k, 0.42, 1)
            wv, _ = tsize(v, 0.42, 1)
            w = wk + wv + 26
            if x + w > x1 - 14:
                break
            rrect(img, x, y, x + w, y + 22, INNER, 11)
            put(img, k, x + 10, y + 16, 0.42, SUB, 1)
            put(img, v, x + 16 + wk, y + 16, 0.42, TXT, 1)
            x += w + 8
        pairs = [p.split(" ", 1) for p in re.split(r"\s{2,}", str(d.get("keys") or "").strip()) if p]
        x, y = x0 + 14, y0 + 58
        for pair in pairs:
            key, action = pair[0], (pair[1] if len(pair) > 1 else "")
            wk, _ = tsize(key, 0.4, 1)
            wa, _ = tsize(action, 0.4, 1)
            if x + wk + wa + 20 > x1 - 10:
                break
            rrect(img, x, y - 14, x + wk + 10, y + 5, EDGE, 4)
            put(img, key, x + 5, y, 0.4, TXT, 1)
            put(img, action, x + wk + 15, y, 0.4, SUB, 1)
            x += wk + wa + 30

    # dynamic parts ---------------------------------------------------------------------------------------------
    def render(self, status, camera_frame=None, twin_panel=None, now=None):
        try:
            return self._render(status or {}, camera_frame, twin_panel, time.monotonic() if now is None else now)
        except Exception:  # a dashboard bug must not stop the evaluation
            err = traceback.format_exc()
            if err != self.last_error:
                self.last_error = err
                print(err, file=sys.stderr)
            img = np.full((self.size[1], self.size[0], 3), BG, np.uint8)
            put(img, "dashboard error: " + err.strip().splitlines()[-1], 20, 40, 0.6, RED, 1)
            return img

    def _render(self, s, cam, twin, now):
        d = s.get("demo") if isinstance(s.get("demo"), dict) else {}
        mode = str(d.get("mode") or "extero")
        names = [str(n) for n in seq(s.get("joint_names"))] or [f"joint_{i}" for i in range(17)]
        has_cam, has_twin = cam is not None, twin is not None
        key = (mode, has_cam, has_twin, tuple(names), d.get("title"), d.get("mode_label"), d.get("model"),
               d.get("task"), repr(d.get("params")), d.get("keys"), self.size)
        if key != self.static_key:
            self.static, self.static_key = self._build_static(s, d, mode, has_cam, has_twin, names), key
        self._record(s, d)
        img = self.static.copy()
        accent = MODE_COLORS.get(mode, BLUE)
        state = str(d.get("state") or ("running" if s.get("running") else "waiting"))

        self._runs(img, d, state, accent, now, s)
        twin_rect, joints_rect = self._layout(mode, has_cam, has_twin)
        if has_cam:
            self._paste(img, cam, MAIN)
        if twin_rect:
            self._paste(img, twin, (twin_rect[0] + 4, twin_rect[1] + 4, twin_rect[2] - 4, twin_rect[3] - 4))
        if joints_rect:
            self._joints(img, d, names)
        self._chart(img, s, d, mode, accent)
        self._state(img, s, d, state)
        self._tiles(img, s, d, mode, names)
        self._motors(img, s, d, names)
        self._overlays(img, s, d, state, accent)
        return img

    def _paste(self, img, frame, rect):
        x0, y0, x1, y1 = rect
        fh, fw = frame.shape[:2]
        if (fw, fh) != (x1 - x0, y1 - y0):
            k = min((x1 - x0) / fw, (y1 - y0) / fh)
            w, h = max(1, int(fw * k)), max(1, int(fh * k))
            if self.twin_cache[0] is not frame:
                self.twin_cache = (frame, cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR))
            frame = self.twin_cache[1]
            x0, y0 = x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2
            x1, y1 = x0 + w, y0 + h
        img[y0:y1, x0:x1] = frame[:, :, :3] if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def _runs(self, img, d, state, accent, now, s):
        runs = count(d.get("runs"))
        if runs <= 0:
            return
        labels, results = seq(d.get("run_labels")), seq(d.get("results"))
        current = count(d.get("run"), 99)
        w = min(150.0, (470 - 10 * (runs - 1)) / runs)
        x = 1584 - runs * w - (runs - 1) * 10
        for i in range(runs):
            r = results[i] if i < len(results) and isinstance(results[i], dict) else None
            live = i + 1 == current and state != "done"
            x0, x1 = int(x), int(x + w)
            if r is not None and not live:
                rrect(img, x0, 14, x1, 70, INNER, 10)
                head, color = str(r.get("headline", "")), TXT
            elif live:
                rrect(img, x0, 14, x1, 70, CARD, 10)
                rrect(img, x0, 14, x1, 70, accent, 10, 2)
                head, color = ("LIVE", GREEN) if state == "running" else (state.upper(), STATE_COLORS.get(state, SUB))
            else:
                rrect(img, x0, 14, x1, 70, EDGE, 10, 1)
                head, color = "-", MUTED
            lab = labels[i] if i < len(labels) else f"run {i + 1}"
            put(img, str(lab).upper(), x0 + 12, 34, 0.42, SUB if r or live else MUTED, 1, max_w=w - 24)
            if live and state == "running":
                cv2.circle(img, (x0 + 18, 54), 5, GREEN if (now % 1.0) < 0.6 else EDGE, -1, AA)
                put(img, head, x0 + 30, 61, 0.6, color, 2, max_w=w - 42)
            else:
                put(img, head, x0 + 12, 61, 0.6, color, 2, max_w=w - 24)
            x += w + 10

    def _joints(self, img, d, names):
        n = len(names)
        q, cmd = nums(d.get("q_deg"), n), nums(d.get("cmd_deg"), n)
        lo, hi = nums(d.get("joint_lo_deg"), n), nums(d.get("joint_hi_deg"), n)
        bx0, bx1 = self.bar_x
        for j, yc, h in self.rows:
            a = lo[j] if lo is not None and np.isfinite(lo[j]) else -30.0
            b = hi[j] if hi is not None and np.isfinite(hi[j]) else 110.0
            b = b if b > a else a + 1

            def px(v):
                return int(bx0 + (bx1 - bx0) * np.clip((v - a) / (b - a), 0, 1))
            if a < 0 < b:
                cv2.line(img, (px(0), yc - h // 2 - 3), (px(0), yc + h // 2 + 3), EDGE, 1)
            if q is not None and np.isfinite(q[j]):
                rrect(img, bx0, yc - h // 2, max(px(q[j]), bx0 + h), yc + h // 2, PURPLE, 3)
                put(img, f"{q[j]:.0f}", bx1 + 60, yc + 5, 0.45, TXT, 1, anchor="r")
            if cmd is not None and np.isfinite(cmd[j]):
                c = px(cmd[j])
                cv2.line(img, (c, yc - h // 2 - 4), (c, yc + h // 2 + 4), YELLOW, 2, AA)

    def _plot(self, img, rect, t, y, duration, ylo, yhi, color, unit="", ref=None, ref_label="", ideal=None,
              label=None, fmt="{:+.1f}", ticks=4):
        """One time series in rect (pixels): grid, y labels, the line, its latest value."""
        x0, y0, x1, y1 = rect
        span = max(yhi - ylo, 1e-6)
        step = nice_step(span, ticks)
        v = math.ceil(ylo / step) * step
        while v <= yhi + 1e-9:
            yy = int(y1 - (v - ylo) / span * (y1 - y0))
            cv2.line(img, (x0, yy), (x1, yy), INNER, 1)
            put(img, f"{v:g}", x0 - 8, yy + 5, 0.4, MUTED, 1, anchor="r")
            v += step
        roi = img[y0 - 2:y1 + 3, x0:x1 + 1]

        def to_px(tt, vv):
            return np.stack([np.clip(tt / duration, -1, 2) * (x1 - x0) * 16, (y1 - y0 + 2 - np.clip((vv - ylo) / span, -1, 2) * (y1 - y0)) * 16],
                            1).astype(np.int32)
        if ref is not None:
            yy = int(y1 - (ref - ylo) / span * (y1 - y0))
            if y0 <= yy <= y1:
                dashed(img, (x0, yy), (x1, yy), RED, 1)
                put(img, ref_label, x1 - 4, yy - 6, 0.4, RED, 1, anchor="r")
        if ideal is not None:
            tt = np.array([0.0, duration])
            pts = to_px(tt, ideal * tt) // 16
            dashed(roi, pts[0], pts[1], SUB, 1, 10, 8)
        if len(t) >= 2:
            ok = np.isfinite(y)
            idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
            for a, b in zip(idx[::2], idx[1::2]):
                if b - a >= 2:
                    cv2.polylines(roi, [to_px(t[a:b], y[a:b])], False, color, 2, AA, 4)
        last = np.flatnonzero(np.isfinite(y))
        if len(last):
            i = last[-1]
            p = to_px(t[i:i + 1], np.clip(y[i:i + 1], ylo, yhi))[0] // 16 + (x0, y0 - 2)
            cv2.circle(img, tuple(int(c) for c in p), 5, color, -1, AA)
            txt = fmt.format(y[i]) + unit
            w, _ = tsize(txt, 0.6, 2)
            tx = int(min(p[0] + 10, x1 - w - 10))
            ty = int(np.clip(p[1] - 10, y0 + 18, y1))
            rrect(img, tx - 6, ty - 19, tx + w + 6, ty + 7, CARD, 6)
            put(img, txt, tx, ty, 0.6, color, 2)
        if label:
            put(img, label, x0 + 6, y0 + 14, 0.42, SUB, 1)

    def _xaxis(self, img, x0, x1, y, duration):
        step = nice_step(duration, 6)
        v = 0.0
        while v <= duration + 1e-9:
            xx = int(x0 + v / duration * (x1 - x0))
            cv2.line(img, (xx, y), (xx, y + 4), MUTED, 1)
            put(img, f"{v:g}{' s' if v + step > duration else ''}", xx, y + 18, 0.4, MUTED, 1, anchor="c")
            v += step

    def _chart(self, img, s, d, mode, accent):
        x0, y0, x1, y1 = CHART
        duration = seconds(s.get("duration"))
        h = {k: np.asarray(v, float) for k, v in self.hist.items()}
        t = h.get("t", np.zeros(0))
        px0, px1, py0, py1 = x0 + 64, x1 - 24, y0 + 46, y1 - 32
        if mode == "proprio":
            limit = num(s.get("temp_limit")) or 65.0
            hot = h.get("hot", np.zeros(0))
            lo = min(30.0, np.nanmin(hot) - 2) if np.isfinite(hot).any() else 30.0
            mid = (py0 + py1) // 2
            self._plot(img, (px0, py0, px1, mid - 10), t, hot, duration, lo, limit + 5, ORANGE, " C", ref=limit,
                       ref_label=f"limit {limit:.0f} C", fmt="{:.0f}", ticks=2)
            tr = h.get("track", np.zeros(0))
            top = max(5.0, np.nanmax(tr) * 1.25) if np.isfinite(tr).any() else 5.0
            self._plot(img, (px0, mid + 14, px1, py1), t, tr, duration, 0, top, PURPLE, " deg", fmt="{:.1f}", ticks=2)
            self._legend(img, x1 - 24, y0 + 28, [(ORANGE, "hottest motor (C)"), (PURPLE, "tracking error (deg)")])
        elif mode == "goals":
            tol = num(d.get("goal_tol_deg"))
            goals = h.get("goals", np.zeros(0))
            for i in np.flatnonzero(np.diff(goals) > 0) + 1 if len(goals) > 1 else []:
                xx = int(px0 + t[i] / duration * (px1 - px0))
                cv2.line(img, (xx, py0), (xx, py1), GREEN, 2, AA)
                cv2.circle(img, (xx, py0), 4, GREEN, -1, AA)
            if tol is not None:
                yy = int(py1 - tol / 180 * (py1 - py0))
                blend(img, (px0, yy, px1, py1), GREEN, 0.12)
                cv2.line(img, (px0, yy), (px1, yy), GREEN, 1, AA)
            self._plot(img, (px0, py0, px1, py1), t, h.get("err", np.zeros(0)), duration, 0, 180, accent, " deg",
                       fmt="{:.0f}")
            legend = [(GREEN, "goal reached"), (GREEN, f"tolerance {tol:.0f} deg" if tol else "")]
            self._legend(img, x1 - 24, y0 + 28, legend)
        else:
            goal = num(s.get("goal"))
            ideal = abs(goal) / (2 * math.pi) if goal else None
            turns = h.get("turns", np.zeros(0))
            el = num(s.get("elapsed")) or 0.0
            vals = [0.0, 1.0] + ([float(np.nanmax(turns)), float(np.nanmin(turns))] if np.isfinite(turns).any()
                                 else []) + ([ideal * min(max(el, 10.0) + 5, duration)] if ideal else [])
            lo, hi = min(vals), max(vals)
            pad = 0.1 * (hi - lo)
            self._plot(img, (px0, py0, px1, py1), t, turns, duration, lo - (pad if lo < 0 else 0), hi + pad, accent,
                       ideal=ideal, fmt="{:+.2f}")
            legend = [(accent, "measured")] + ([(SUB, f"ideal at {abs(goal):.1f} rad/s")] if ideal else [])
            self._legend(img, x1 - 24, y0 + 28, legend)
        self._xaxis(img, px0, px1, py1 + 4, duration)

    def _legend(self, img, x, y, items):
        for color, label in reversed([i for i in items if i[1]]):
            w = put(img, label, x, y, 0.42, SUB, 1, anchor="r")
            cv2.line(img, (int(x - w - 30), y - 5), (int(x - w - 10), y - 5), color, 2, AA)
            x -= w + 50

    def _state(self, img, s, d, state):
        x0, y0, x1, y1 = STATE_CARD
        color = STATE_COLORS.get(state, SUB)
        put(img, "STATE", x0 + 18, y0 + 28, 0.5, SUB, 1)
        cv2.circle(img, (x0 + 27, y0 + 62), 8, color, -1, AA)
        put(img, state.upper(), x0 + 46, y0 + 74, 1.15, color, 2, max_w=300)
        phase = str(s.get("phase") or "")
        put(img, phase, x0 + 18, y0 + 102, 0.48, SUB, 1, max_w=330)
        duration = seconds(s.get("duration"))
        el = float(np.clip(num(s.get("elapsed")) or 0.0, 0, duration))
        put(img, "REMAINING", x1 - 18, y0 + 28, 0.5, SUB, 1, anchor="r")
        w = put(img, "s", x1 - 18, y0 + 98, 0.8, SUB, 2, anchor="r")
        put(img, f"{math.ceil(duration - el - 1e-6):d}", x1 - 26 - w, y0 + 98, 2.4, TXT, 3, NUM, anchor="r")
        bx0, bx1, by = x0 + 18, x1 - 18, y0 + 114
        rrect(img, bx0, by, bx1, by + 8, INNER, 4)
        if el > 0:
            rrect(img, bx0, by, max(bx0 + 8, int(bx0 + (bx1 - bx0) * el / duration)), by + 8, color, 4)

    def _tile(self, img, i, value, unit="", sub="", color=TXT):
        x0, y0, x1, y1 = TILES[i]
        font, scale = NUM, 1.6
        if value == "-":
            value, color, font, scale = "--", MUTED, SANS, 1.4
        w = put(img, value, x0 + 18, y0 + 78, scale, color, 2, font, max_w=x1 - x0 - 110)
        if unit:
            put(img, unit, x0 + 26 + w, y0 + 78, 0.6, SUB, 1, max_w=x1 - x0 - 40 - w)
        if sub:
            put(img, sub, x0 + 18, y0 + 101, 0.45, MUTED, 1, max_w=x1 - x0 - 36)

    def _tiles(self, img, s, d, mode, names):
        temps = nums(s.get("temps"))
        limit = num(s.get("temp_limit")) or 65.0
        start = num(d.get("start_temp_c")) or limit - 25
        dash = "-"
        if mode == "goals":
            goals, err, tol = num(s.get("goals")), num(s.get("goal_error_deg")), num(d.get("goal_tol_deg"))
            el = num(s.get("elapsed"))
            self._tile(img, 0, dash if goals is None else f"{goals:.0f}", "goals", "", AMBER)
            self._tile(img, 1, dash if err is None else f"{err:.0f}", "deg", f"reached below {tol:.0f} deg" if tol else "",
                       GREEN if err is not None and tol and err < tol else TXT)
            self._tile(img, 2, f"{el / goals:.1f}" if goals and el else dash, "s", "policy time per goal")
        elif mode == "proprio":
            loop = d.get("loop") if isinstance(d.get("loop"), dict) else {}
            hz, jit, inf = num(loop.get("hz")), num(loop.get("jitter_ms")), num(loop.get("infer_ms"))
            sub = "  ".join(x for x in (f"jitter {jit:.1f} ms" if jit is not None else "",
                                        f"inference {inf:.1f} ms" if inf is not None else "") if x)
            self._tile(img, 0, dash if hz is None else f"{hz:.1f}", "Hz", sub)
            tr = num(d.get("track_err_deg"))
            self._tile(img, 1, dash if tr is None else f"{tr:.1f}", "deg", "mean over joints, command vs measured")
            cur = nums(d.get("current_ma"))
            if cur is not None and np.isfinite(cur).any():
                j = int(np.nanargmax(np.abs(cur)))
                self._tile(img, 2, f"{abs(cur[j]):.0f}", "mA", names[j] if j < len(names) else "",
                           RED if abs(cur[j]) > 800 else TXT)
            else:
                self._tile(img, 2, dash, "mA")
        else:
            turns, speed, goal = num(s.get("turns")), num(s.get("speed")), num(s.get("goal"))
            self._tile(img, 0, dash if turns is None else f"{turns:+.2f}", "turns",
                       f"about {d['axis']}" if d.get("axis") else "+ = commanded direction", MODE_COLORS.get(mode, BLUE))
            ok = speed is not None and goal and speed * goal > 0 and abs(speed) >= 0.5 * abs(goal)
            self._tile(img, 1, dash if speed is None else f"{speed:+.1f}",
                       f"/ {goal:+.1f} rad/s" if goal is not None else "rad/s", "", GREEN if ok else TXT)
            hz = num(d.get("detect_hz"))
            self._tile(img, 2, dash if hz is None else f"{hz:.0f}", "poses/s",
                       "camera + pose network" if d.get("camera", True) else "no camera",
                       MUTED if hz is None else GREEN if hz >= 10 else ORANGE if hz >= 3 else RED)
        if temps is not None and np.isfinite(temps).any():
            j = int(np.nanargmax(temps))
            self._tile(img, 3, f"{temps[j]:.0f}", "C", f"{names[j] if j < len(names) else ''}  (limit {limit:.0f} C)",
                       temp_color(temps[j], start, limit))
        else:
            self._tile(img, 3, dash, "C", "no temperatures")

    def _motors(self, img, s, d, names):
        n = len(names)
        temps, cur = nums(s.get("temps"), n), nums(d.get("current_ma"), n)
        limit = num(s.get("temp_limit")) or 65.0
        start = num(d.get("start_temp_c"))
        x0, y0, x1, y1 = self.temp_plot
        tmax = max(limit + 5, float(np.nanmax(temps)) + 3 if temps is not None and np.isfinite(temps).any() else 0)
        tmin = 20.0

        def ty(v):
            return int(y1 - (np.clip(v, tmin, tmax) - tmin) / (tmax - tmin) * (y1 - y0))
        mx0, _, mx1, _ = MOTORS
        legend = f"start below {start:.0f} C   limit {limit:.0f} C" if start is not None else f"limit {limit:.0f} C"
        put(img, legend, mx1 - 18, MOTORS[1] + 28, 0.42, SUB, 1, anchor="r")
        for j, a, b in self.mbars:
            cv2.rectangle(img, (a, y0), (b, y1), INNER, -1)
        if start is not None:
            ys = ty(start)
            dashed(img, (x0 - 4, ys), (x1 + 4, ys), YELLOW, 1, 6, 5)
            put(img, f"{start:.0f}", x1 + 10, ys + 5, 0.45, YELLOW, 1)
        for j, a, b in self.mbars:
            t = temps[j] if temps is not None else np.nan
            if np.isfinite(t):
                cv2.rectangle(img, (a, ty(t)), (b, y1), temp_color(t, start or limit - 25, limit), -1)
        yl = ty(limit)
        cv2.line(img, (x0 - 4, yl), (x1 + 4, yl), RED, 2, AA)
        put(img, f"{limit:.0f}", x1 + 10, yl + 5, 0.45, RED, 1)
        for j, a, b in self.mbars:
            t = temps[j] if temps is not None else np.nan
            if np.isfinite(t):
                cv2.rectangle(img, (a - 2, ty(t) - 17), (b + 2, ty(t) - 2), CARD, -1)
                put(img, f"{t:.0f}", (a + b) / 2, ty(t) - 5, 0.4, TXT, 1, anchor="c")
        c0, c1 = self.cur_plot
        if cur is not None:
            for j, a, b in self.mbars:
                if np.isfinite(cur[j]):
                    v = min(abs(cur[j]), 1000) / 1000
                    cv2.rectangle(img, (a, c1), (b, int(c1 - v * (c1 - c0))), RED if v > 0.8 else BLUE, -1)
            put(img, f"peak {np.nanmax(np.abs(cur)):.0f} mA   scale 0..1000 mA" if np.isfinite(cur).any() else "",
                MOTORS[2] - 18, c0 - 10, 0.42, SUB, 1, anchor="r")
        cv2.line(img, (x0, c1), (x1, c1), EDGE, 1)

    def _overlays(self, img, s, d, state, accent):
        x0, y0, x1, y1 = MAIN
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if state == "paused":
            blend(img, MAIN, BG, 0.55)
            shadow(img, "PAUSED", cx, cy + 30, 3.0, WHITE, 6, NUM)
        elif state == "stopped":
            blend(img, MAIN, RED, 0.35)
            shadow(img, "TORQUE OFF", cx, cy + 30, 2.8, WHITE, 6, NUM)
        elif state == "countdown":
            c = num(d.get("countdown"))
            if c is not None:
                roi = img[cy - 150:cy + 150, cx - 150:cx + 150]
                disk = roi.copy()
                cv2.circle(disk, (150, 150), 140, BG, -1, AA)
                cv2.addWeighted(disk, 0.7, roi, 0.3, 0, dst=roi)
                cv2.circle(img, (cx, cy), 128, EDGE, 8, AA)
                frac = c - math.floor(c) if c > 0 else 0
                cv2.ellipse(img, (cx, cy), (128, 128), -90, 0, 360 * frac, YELLOW, 8, AA)
                n = str(max(1, math.ceil(c - 1e-6)))
                _, th = tsize(n, 6.0, 12, NUM)
                put(img, n, cx, cy + th // 2, 6.0, WHITE, 12, NUM, anchor="c")
        elif state == "done":
            self._summary(img, d, accent)
        banner = s.get("banner")
        if banner and state != "done":
            color = COLORS.get(str(s.get("banner_color")), YELLOW)
            scale = 1.05
            w, th = tsize(str(banner), scale, 2)
            scale *= min(1.0, (x1 - x0 - 120) / max(w, 1))
            w, th = tsize(str(banner), scale, 2)
            by0, by1 = y1 - 110, y1 - 30
            blend(img, (x0 + 24, by0, x1 - 24, by1), BG, 0.82)
            cv2.rectangle(img, (x0 + 24, by0), (x0 + 32, by1), color, -1)
            put(img, str(banner), cx + 4, (by0 + by1) // 2 + th // 2, scale, color, 2, anchor="c")

    def _summary(self, img, d, accent):
        x0, y0, x1, y1 = MAIN
        blend(img, MAIN, BG, 0.8)
        runs = count(d.get("runs"))
        labels, results = seq(d.get("run_labels")), seq(d.get("results"))
        cx = (x0 + x1) // 2
        put(img, "RESULTS", cx, y0 + 70, 1.0, SUB, 2, anchor="c")
        put(img, str(d.get("model") or "").upper() + ("  -  " + str(d.get("task")) if d.get("task") else ""), cx,
            y0 + 110, 0.75, TXT, 2, anchor="c", max_w=x1 - x0 - 80)
        row = min(110, (y1 - y0 - 170) // max(runs, 1))
        for i in range(runs):
            r = results[i] if i < len(results) and isinstance(results[i], dict) else {}
            ry = y0 + 150 + i * row
            rrect(img, x0 + 60, ry, x1 - 60, ry + row - 16, CARD, 12)
            cv2.rectangle(img, (x0 + 60, ry + 10), (x0 + 66, ry + row - 26), accent, -1)
            mid = ry + (row - 16) // 2
            put(img, str(labels[i] if i < len(labels) else f"run {i + 1}").upper(), x0 + 90, mid + 10, 0.8, SUB, 2,
                max_w=200)
            put(img, str(r.get("headline") or "-"), x0 + 310, mid + 18, 1.5, TXT if r else MUTED, 2, NUM, max_w=330)
            put(img, str(r.get("detail") or ""), x1 - 90, mid + 10, 0.7, SUB, 1, anchor="r", max_w=220)


# self-test ---------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    out = sys.argv[1] if len(sys.argv) > 1 else "/Users/davidhohenstatt/.claude/jobs/94b62cff/tmp/dash"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    NAMES = ["wrist", "index_abd", "middle_abd", "pinky_abd", "ring_abd", "thumb_cmc", "index_mcp", "middle_mcp",
             "pinky_mcp", "ring_mcp", "thumb_abd", "index_pip", "middle_pip", "pinky_pip", "ring_pip", "thumb_mcp",
             "thumb_dip"]
    LO = [-65, -25, -27, -30, -27, -45, -25, -25, -25, -25, -18, -15, -15, -15, -15, -25, -15]
    HI = [35, 30, 27, 30, 27, 33, 100, 100, 100, 100, 55, 107, 107, 107, 107, 100, 107]
    cam = np.zeros((540, 960, 3), np.uint8)
    cam[:] = np.linspace(40, 90, 960, dtype=np.uint8)[None, :, None]
    cv2.circle(cam, (480, 300), 170, (120, 130, 140), -1)
    cv2.polylines(cam, [np.array([[430, 230], [540, 250], [520, 360], [410, 340]])], True, (80, 220, 80), 3, AA)
    twin = np.full((380, 300, 3), 48, np.uint8)
    cv2.ellipse(twin, (150, 230), (80, 110), 0, 0, 360, (190, 188, 184), -1, AA)
    for k in range(5):
        cv2.line(twin, (90 + 30 * k, 160), (80 + 35 * k, 60), (190, 188, 184), 16, AA)

    def status(mode, t, state="running", **extra):
        goal = 1.5
        demo = dict(title="ORCA Hand - in-hand cube rotation", mode=mode, model="ibex",
                    mode_label=dict(proprio="PROPRIOCEPTIVE", extero="EXTEROCEPTIVE (camera)",
                                    axes="OPTIONAL - ALL AXES", goals="OPTIONAL - GOAL ORIENTATIONS")[mode],
                    task="reach goal orientations" if mode == "goals" else "spin +1.5 rad/s about z",
                    axis="y" if mode == "axes" else "z", goal_tol_deg=23 if mode == "goals" else None, run=2, runs=3,
                    run_labels=["about x", "about y", "about z"] if mode == "axes" else ["run 1", "run 2", "run 3"],
                    results=[{"headline": "+3.2 turns", "detail": "1.6 rad/s avg"}, None, None], state=state,
                    countdown=None, camera=mode != "proprio", start_temp_c=38.0, joint_lo_deg=LO, joint_hi_deg=HI,
                    q_deg=[(lo + hi) / 2 + 0.4 * (hi - lo) * math.sin(t * 2 + i) for i, (lo, hi) in enumerate(zip(LO, HI))],
                    cmd_deg=[(lo + hi) / 2 + 0.4 * (hi - lo) * math.sin(t * 2 + i + 0.3) for i, (lo, hi) in
                             enumerate(zip(LO, HI))],
                    current_ma=list(rng.uniform(50, 900, 17)), voltage_v=12.1,
                    loop={"hz": 19.98, "jitter_ms": 1.1, "infer_ms": 0.7}, track_err_deg=3 + math.sin(t) + rng.normal(0, .3),
                    detect_hz=28.4 if mode != "proprio" else None,
                    params=[["servo acc", "150"], ["action alpha", "0.2"], ["speed", "+1.5 rad/s"],
                            ["start below", "38 C"], ["limit", "65 C"]],
                    keys="SPACE start   P pause   E stop   N skip run   Q quit   F fullscreen")
        temps = [34 + 0.5 * t * (i % 5) / 4 + i % 3 for i in range(17)]
        st = dict(phase=f"run 2 of 3 - {state}", banner=None, banner_color="green", elapsed=t, duration=60,
                  running=state == "running", turns=0.18 * t + 0.05 * math.sin(t), speed=1.2 + 0.1 * math.sin(t), goal=goal,
                  goals=int(t // 6) if mode == "goals" else None,
                  goal_error_deg=(150 - 25 * (t % 6) if (t % 6) < 5.5 else None) if mode == "goals" else None,
                  temps=temps, temp_limit=65, joint_names=NAMES, trail_id=7, demo=demo)
        st.update({k: v for k, v in extra.items() if k != "demo"})
        demo.update(extra.get("demo", {}))
        return st

    cases = [("proprio", None, twin, {}), ("extero", cam, twin, {}), ("axes", cam, None, {}), ("goals", cam, None, {})]
    times = []
    for mode, c, tw, extra in cases:
        dash = Dashboard()
        for k in range(400):  # 20 s at 20 Hz
            st = status(mode, k * 0.05)
            t0 = time.perf_counter()
            img = dash.render(st, c, tw, now=k * 0.05)
            times.append(time.perf_counter() - t0)
        cv2.imwrite(f"{out}/{mode}.png", img)
    dash = Dashboard()
    for k in range(200):
        dash.render(status("extero", k * 0.05), cam, None)
    specials = {
        "countdown": status("extero", 0.0, "countdown", banner="START IN 3 - hands away", banner_color="yellow",
                            demo=dict(countdown=2.4)),
        "cooling": status("extero", 0.0, "cooling", banner="COOLING: thumb_mcp 47 C, start below 38 C",
                          banner_color="orange"),
        "waiting": status("proprio", 0.0, "waiting", banner="Place the cube in the palm, then press SPACE",
                          banner_color="yellow", demo=dict(q_deg=None, cmd_deg=None, current_ma=None, loop=None)),
        "paused": status("extero", 10.0, "paused"),
        "stopped": status("goals", 10.0, "stopped", banner="E pressed: torque off", banner_color="red"),
        "done": status("extero", 60.0, "done", demo=dict(run=3, results=[
            {"headline": "+3.2 turns", "detail": "1.6 rad/s avg"}, {"headline": "+2.7 turns", "detail": "1.4 rad/s avg"},
            {"headline": "+3.0 turns", "detail": "1.5 rad/s avg"}])),
        "empty": {},
        "partial": dict(phase="starting", demo=dict(mode="axes", state="starting"), temps=[None] * 17, elapsed=float("nan")),
    }
    for name, st in specials.items():
        t0 = time.perf_counter()
        img = dash.render(st, None if name in ("waiting", "empty", "partial") else cam,
                          twin if name == "waiting" else None)
        times.append(time.perf_counter() - t0)
        cv2.imwrite(f"{out}/{name}.png", img)
    times = np.array(times[20:]) * 1000
    print(f"render: mean {times.mean():.2f} ms, p95 {np.percentile(times, 95):.2f} ms, max {times.max():.2f} ms "
          f"({len(times)} frames, incl. static rebuilds); PNGs in {out}")
