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


def save_code_changes(repo: Path, path: Path) -> str | None:
    """git diff of the uncommitted changes (tracked files) plus the content of new files: with the commit, this is
    the code that ran. Returns the file name, or None without changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout
    try:
        diff = git("diff", "HEAD")
        untracked = [f for f in git("ls-files", "--others", "--exclude-standard").splitlines()
                     if f.endswith((".py", ".yaml", ".yml", ".sh", ".xml", ".md", ".toml"))]
    except OSError:
        return None
    for f in untracked:
        try:
            diff += f"\n=== new file (untracked): {f}\n" + (repo / f).read_text(errors="replace")
        except OSError:
            pass
    if not diff.strip():
        return None
    path.write_text(diff)
    return path.name


def session_info(args, repo: Path, hand_config: Path, out: Path | None = None) -> dict:
    """What stays the same for all runs of one evaluation."""
    started = datetime.now()
    changes = save_code_changes(repo, out / f"code_changes_{started:%Y%m%d_%H%M%S}.diff") if out else None
    return dict(host=socket.gethostname(), started=started.isoformat(timespec="seconds"),
                options=vars(args), note=getattr(args, "note", ""), rwr_system=git_state(repo) | dict(
                    uncommitted_changes_file=changes),
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
