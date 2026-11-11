"""The evaluation's vision process: camera, cube detector, live window and video. No ROS.

It runs in a process of its own (run() is the target), so that neither the detector nor the video encoding can
delay the 20 Hz policy loop, and so that the window has a main thread (macOS needs that).

    status_q   in:  the newest status dict from eval_policies.py (what the panel shows, where to record)
    events_q   out: ("ready", info) once, then ("det", time, position, quaternion xyzw, centre pixel),
                    ("key", character), ("imu", acceleration xyz, m/s^2, about once per second), ("error", text)

The camera is read directly with depthai (the OAK's own exposure, focus and IMU are then available); --mock
replaces camera and detector with a drawn cube that turns while the policy runs.
"""

import queue
import signal
import threading
import time
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .hud import project, render

WINDOW = "cube spin evaluation"


class OakCamera:
    """The OAK's colour camera at 960x540 with fixed focus and exposure, and its accelerometer."""

    def __init__(self, exposure_us=5000, iso=400, fps=30, camera_id=""):
        import depthai as dai
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setIspScale(1, 2)
        cam.setFps(fps)
        cam.initialControl.setManualFocus(130)  # as the OAK-D ROS node: sharp at the hand's distance
        if exposure_us > 0:
            cam.initialControl.setManualExposure(int(exposure_us), int(iso))
        out = pipeline.createXLinkOut()
        out.setStreamName("color")
        cam.isp.link(out.input)
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
        imu.setBatchReportThreshold(10)
        imu.setMaxBatchReports(20)
        imu_out = pipeline.createXLinkOut()
        imu_out.setStreamName("imu")
        imu.out.link(imu_out.input)

        self.device = dai.Device(pipeline, dai.DeviceInfo(camera_id)) if camera_id else dai.Device(pipeline)
        self.queue = self.device.getOutputQueue("color", maxSize=2, blocking=False)
        self.imu_queue = self.device.getOutputQueue("imu", maxSize=20, blocking=False)
        calib = self.device.readCalibration()
        self.K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 960, 540))
        self.dist = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
        self.info = dict(camera=self.device.getDeviceName(), camera_id=self.device.getMxId(),
                         usb=str(self.device.getUsbSpeed()).split(".")[-1], exposure_us=exposure_us, iso=iso)

    def read(self):
        """The newest image (BGR) or None."""
        msg = self.queue.tryGet()
        return None if msg is None else msg.getCvFrame()

    def acceleration(self):
        """Mean of the accelerometer readings since the last call (m/s^2, the IMU's axes), or None."""
        samples = []
        while True:
            msg = self.imu_queue.tryGet()
            if msg is None:
                break
            samples += [[p.acceleroMeter.x, p.acceleroMeter.y, p.acceleroMeter.z] for p in msg.packets]
        return np.mean(samples, axis=0) if samples else None

    def close(self):
        self.device.close()


class MockCamera:
    """A drawn cube for tests without hardware. It turns at ``rate`` while the status says the policy runs, and
    falls out of the hand once, ``drop_after`` seconds into the run, until the status asks for the cube back."""

    K = np.array([[773.0, 0.0, 480.0], [0.0, 773.0, 270.0], [0.0, 0.0, 1.0]])
    dist = None
    info = dict(camera="mock", camera_id="mock", usb="-", exposure_us=0, iso=0)
    HOME = np.array([0.02, 0.01, 0.35])

    def __init__(self, rate=1.6, drop_after=None, fps=30):
        self.rate, self.drop_after, self.fps = rate, drop_after, fps
        self.yaw, self.last, self.away_since, self.dropped = 0.0, time.time(), None, False
        self.status = {}

    def read(self):
        time.sleep(1.0 / self.fps)
        now = time.time()
        s = self.status
        if s.get("running"):
            self.yaw += self.rate * np.sign(s.get("goal", 1.0)) * (now - self.last)
            if self.drop_after is not None and not self.dropped and s.get("elapsed", 0.0) >= self.drop_after:
                self.dropped, self.away_since = True, now
        elif self.away_since is not None and s.get("want_cube") and now - self.away_since > 3.0:
            self.away_since = None  # "a person" puts it back
        self.last = now
        position = self.HOME + (np.array([0.14, 0.05, 0.03]) if self.away_since is not None else 0.0)
        self.pose = (Rotation.from_rotvec([0.0, 0.0, -self.yaw]).as_matrix(), position)
        image = np.full((540, 960, 3), 60, np.uint8)
        signs = np.array([[x, y, -1] for x, y in ((1, 1), (-1, 1), (-1, -1), (1, -1))], dtype=float)
        top = project(signs * 0.0225 @ self.pose[0].T + position, self.K)
        cv2.fillPoly(image, [top.astype(np.int32)], (255, 60, 0))
        cv2.line(image, tuple(top[0].astype(int)), tuple(top[2].astype(int)), (255, 255, 255), 3)
        return image

    def detect(self, image, K, dist):
        R, t = self.pose
        return R, t + np.random.normal(0, 0.0005, 3), None, 1.0

    def acceleration(self):
        return np.array([0.0, 0.0, 9.81])

    def close(self):
        pass


class Recorder:
    """Writes the newest frame at a fixed rate from its own thread, so the video runs in real time whatever the
    camera and the encoder do. pause(True) stops the clock: nothing is written, and the video has no gap."""

    def __init__(self, path, fps=30.0):
        self.path, self.fps = str(path), fps
        self.frames = 0
        self.ok = True
        self._frame, self._paused = None, False
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def write(self, frame):
        with self._lock:
            self._frame = frame

    def pause(self, paused):
        self._paused = paused

    def _loop(self):
        writer, next_time = None, None
        try:
            while not self._stop.wait(0.2 / self.fps):
                with self._lock:
                    frame = self._frame
                if frame is None or self._paused:
                    next_time = None
                    continue
                if writer is None:
                    h, w = frame.shape[:2]
                    for codec in ("avc1", "mp4v"):  # H.264 where this OpenCV has it
                        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
                        if writer.isOpened():
                            break
                    if not writer.isOpened():
                        self.ok = False
                        return
                now = time.time()
                next_time = now if next_time is None else next_time
                if now - next_time > 1.0:  # the encoder fell behind: skip instead of piling up
                    next_time = now
                while next_time <= now:
                    writer.write(frame)
                    self.frames += 1
                    next_time += 1.0 / self.fps
        finally:
            if writer is not None:
                writer.release()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=10)


def run(config, status_q, events_q, stop):
    """The process: config is a dict (mock, mock_drop_after, exposure_us, iso, camera_id, detector_ckpt, device,
    cube_edge, window, trail_s)."""
    # Ctrl+C in the terminal also reaches this process: ignore it, so that the video is closed properly when the
    # main process asks this one to stop
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _run(config, status_q, events_q, stop)
    except Exception as e:  # the main process decides what to do
        import traceback
        events_q.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _run(config, status_q, events_q, stop):
    if config.get("mock"):
        camera = MockCamera(drop_after=config.get("mock_drop_after"))
        detector = camera
    else:
        from .detector import CubePoseEstimator
        detector = CubePoseEstimator(config["detector_ckpt"], config.get("cube_edge", 0.045), config.get("device", "auto"))
        camera = OakCamera(config.get("exposure_us", 5000), config.get("iso", 400), camera_id=config.get("camera_id", ""))
    K, dist = camera.K, camera.dist
    events_q.put(("ready", dict(camera.info, K=K.tolist(), detector_device=str(getattr(detector, "device", "mock")))))

    lock = threading.Lock()
    shared = dict(image=None, image_time=0.0, image_id=0, detection=None, detection_time=None)
    detection_times = deque(maxlen=50)
    trail = deque()  # (time, pixel) of the cube's centre

    def detect_loop():
        done = 0
        while not stop.is_set():
            with lock:
                image, image_id, stamp = shared["image"], shared["image_id"], shared["image_time"]
            if image is None or image_id == done:
                time.sleep(0.003)
                continue
            done = image_id
            result = detector.detect(image, K, dist)
            if result is None:
                continue
            R, t, corners, _ = result
            centre = project(t, K, dist)
            pixel = (float("nan"), float("nan")) if centre is None else (float(centre[0][0]), float(centre[0][1]))
            with lock:
                shared["detection"], shared["detection_time"] = (R, t, corners), stamp
                detection_times.append(stamp)
                if centre is not None:
                    trail.append((stamp, pixel))
            events_q.put(("det", stamp, t.tolist(), Rotation.from_matrix(R).as_quat().tolist(), pixel))

    threading.Thread(target=detect_loop, daemon=True).start()

    show = config.get("window", True)
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 540)
    status, recorder, record_path, trail_id = {}, None, None, None
    last_imu = 0.0
    while not stop.is_set():
        try:
            while True:
                status = status_q.get_nowait()
        except queue.Empty:
            pass
        if status.get("quit"):
            break
        if config.get("mock"):
            camera.status = status
        image = camera.read()
        now = time.time()
        if image is None:
            time.sleep(0.002)
            continue
        with lock:
            shared["image"], shared["image_time"] = image, now
            shared["image_id"] += 1
            detection, detection_time = shared["detection"], shared["detection_time"]
            if status.get("trail_id") != trail_id:  # a new run: the path starts anew
                trail_id = status.get("trail_id")
                trail.clear()
            while trail and trail[0][0] < now - config.get("trail_s", 8.0):
                trail.popleft()
            pixels = [p for _, p in trail]
            hz = sum(1 for t in detection_times if now - t <= 2.0) / 2.0
        frame = render(image, status, detection, pixels, K, dist, config.get("cube_edge", 0.045),
                       None if detection_time is None else now - detection_time, hz)

        path = status.get("record")
        if path != record_path:
            if recorder is not None:
                recorder.close()
                if not recorder.ok:
                    events_q.put(("error", f"could not write the video {record_path}"))
            recorder = Recorder(path) if path else None
            record_path = path
        if recorder is not None:
            recorder.pause(not status.get("record_on", True))
            recorder.write(frame)

        if show:
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                events_q.put(("key", chr(key)))
        if now - last_imu >= 1.0:
            last_imu = now
            acc = camera.acceleration()
            if acc is not None:
                events_q.put(("imu", acc.tolist()))
    if recorder is not None:
        recorder.close()
    camera.close()
    if show:
        cv2.destroyAllWindows()
