"""meta.json of a run: what was run and how, so that a result can be reproduced and explained later."""

import hashlib
import json
import socket
import subprocess
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()
    try:
        return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                    changed_files=git("status", "--porcelain").splitlines())
    except OSError:
        return {}


def session_info(args, repo: Path, hand_config: Path) -> dict:
    """What stays the same for all runs of one evaluation."""
    return dict(host=socket.gethostname(), started=datetime.now().isoformat(timespec="seconds"),
                options=vars(args), rwr_system=git_state(repo),
                hand_config=dict(path=str(hand_config), sha256=sha256(hand_config)))


def write_meta(run_dir: Path, session: dict, model: dict, speed: float, start_temps: dict, result: dict | None = None):
    policy = Path(model["dir"]) / "policy.pt"
    meta = dict(session=session,
                model={k: v for k, v in model.items() if k != "dir"} | dict(
                    policy=str(policy), policy_sha256=sha256(policy) if policy.exists() else None),
                goal_ang_vel=speed, start_temperatures_c=start_temps, result=result)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


def update_meta_result(run_dir: Path, result: dict):
    path = run_dir / "meta.json"
    if path.exists():
        meta = json.loads(path.read_text())
        meta["result"] = result
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
