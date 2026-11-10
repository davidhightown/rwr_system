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
        log = open(self.log_dir / f"{log_name or name}.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
