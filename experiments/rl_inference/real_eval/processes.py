"""Start and stop the ROS nodes of the evaluation as separate process groups, with their output in log files."""

import os
import signal
import subprocess
import time
from pathlib import Path


class Processes:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.running: dict[str, tuple[subprocess.Popen, object]] = {}

    def start(self, name: str, cmd: list[str], log_name: str | None = None) -> subprocess.Popen:
        # append: a restarted node must not overwrite the log of the one that stopped
        log = open(self.log_dir / f"{log_name or name}.log", "a")
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} started: {' '.join(cmd)}\n")
        log.flush()
        # setpriv --pdeathsig: the node gets Ctrl+C when this process dies, also when it crashes without clean-up
        # (otherwise the camera and the hand controller keep running and hold the camera and the motor port)
        proc = subprocess.Popen(["setpriv", "--pdeathsig", "INT", "--", *cmd], stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.running[name] = (proc, log)
        return proc

    def alive(self, name: str) -> bool:
        return name in self.running and self.running[name][0].poll() is None

    def stop(self, name: str, timeout: float = 8.0):
        """Stop a process group like Ctrl+C, then harder if it does not exit."""
        if name not in self.running:
            return
        proc, log = self.running.pop(name)
        for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
            if proc.poll() is not None:
                break
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        log.close()

    def stop_all(self):
        for name in reversed(list(self.running)):
            self.stop(name)


def wait_or_fail(check, timeout: float, interval: float = 0.2) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if check():
            return True
        time.sleep(interval)
    return False


# the nodes this evaluation starts (also from an earlier session whose evaluation crashed or was killed)
NODE_PATTERNS = ("cube_spin_v2.launch.py", "oakd_node.py", "cube_detection_node.py",
                 "orcahand_v2_controller_launch.yaml", "hand_control_node.py", "visualize_joints.py",
                 "policy_inference_cube_spin_v2.py", "rqt_image_view /real_eval/view")


def find_leftovers() -> dict[int, str]:
    """{pid: command line} of running nodes of the evaluation that are not children of this process."""
    found = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if any(p in cmd for p in NODE_PATTERNS) and "evaluate_real_policies" not in cmd:
            found[int(proc.name)] = cmd
    return found


def stop_pids(pids, timeout: float = 8.0):
    """Ctrl+C to the process groups of these processes, then harder for those that do not exit."""
    groups = set()
    for pid in pids:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
        for g in list(groups):
            try:
                os.killpg(g, sig)
            except ProcessLookupError:
                groups.discard(g)
        t0 = time.time()
        while groups and time.time() - t0 < wait:
            groups &= _live_groups()
            time.sleep(0.2)
        if not groups:
            return True
    return not (groups & _live_groups())


def _live_groups() -> set[int]:
    """Process groups with at least one process that has not exited (zombies do not count)."""
    live = set()
    for proc in Path("/proc").iterdir():
        if proc.name.isdigit():
            try:
                fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if fields[0] != "Z":
                live.add(int(fields[2]))  # state, ppid, pgrp
    return live
