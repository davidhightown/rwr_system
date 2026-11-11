"""The saved runs of evaluate_real_policies.py and which of them can be compared, for overview.py and upload_wandb.py.

Standard library only. A run counts when it was saved (valid = yes in its session's results.csv). Two conditions
decide whether runs can be compared: the commanded speed (at the model's own goal or not) and the start temperature
(cold start: every motor below MAX_START_C). The leaderboard and OVERVIEW.md use the runs with both; W&B gets the
conditions themselves, to filter on. The policy runs in its own process, so the evaluation's code changes did not change
the turns, only what is logged.
"""

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

MAX_START_C = 42.5
REFERENCE_MODEL = "48_ibex"  # run at the start and end of a session: shows whether the hand changed


@dataclass
class SavedRun:
    session: Path  # the session folder (eval_<date_time>)
    folder: Path  # the run folder
    row: dict  # its row in the session's results.csv
    own_goal: float | None  # the model's goal speed from models.json
    not_comparable_because: str  # empty: at its own goal speed and a cold start

    @property
    def comparable(self) -> bool:
        return not self.not_comparable_because

    @property
    def goal(self) -> float | None:
        return number(self.row.get("goal_ang_vel"))

    @property
    def at_own_goal(self) -> bool:
        return self.own_goal is None or self.goal is None or abs(self.goal - self.own_goal) <= 1e-6

    @property
    def start_temp(self) -> float | None:
        return number(self.row.get("start_temp_max"))

    @property
    def cold_start(self) -> bool:
        return self.start_temp is None or self.start_temp <= MAX_START_C

    @property
    def model(self) -> str:
        return self.row["model"]

    @property
    def run_id(self) -> str:
        """The same on every call and after moving the session folder (e.g. into archive/)."""
        key = f"{self.session.name}/{self.folder.name}"
        name = re.sub(r"[^a-z0-9_-]", "-", f"{self.session.name[5:]}-{self.folder.name}".lower())[:56]
        return f"{name}-{hashlib.sha1(key.encode()).hexdigest()[:6]}"


def number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def load_models(models_dir: Path) -> dict:
    """{"48_ibex": entry of models.json}"""
    return {f"{m['idx']:02d}_{m['name']}": m for m in json.load(open(Path(models_dir) / "models.json"))}


def run_folder(session: Path, row: dict) -> Path:
    """The run's folder, also when the session was moved after the run."""
    path = Path(row.get("run_dir") or "")
    if path.name and path.is_dir():
        return path
    if path.name and (session / path.name).is_dir():
        return session / path.name
    old = row.get("video") or row.get("policy_log") or ""  # results.csv of the first version (9 Nov)
    return session / Path(old).parent.name if old else session


def not_comparable(row: dict, own_goal: float | None) -> str:
    reasons = []
    goal = number(row.get("goal_ang_vel"))
    if own_goal is not None and goal is not None and abs(goal - own_goal) > 1e-6:
        reasons.append(f"goal {goal:g} instead of {own_goal:g} rad/s")
    start = number(row.get("start_temp_max"))
    if start is not None and start > MAX_START_C:
        reasons.append(f"started at {start:.0f} C")
    return "; ".join(reasons)


def saved_runs(models_dir: Path) -> list[SavedRun]:
    """Every saved run of all sessions, also those in archive/, oldest first."""
    models_dir = Path(models_dir)
    models = load_models(models_dir)
    runs = []
    files = sorted(models_dir.glob("eval_*/results.csv")) + sorted(models_dir.glob("archive/eval_*/results.csv"))
    for results in files:
        session = results.parent
        for row in csv.DictReader(open(results)):
            if row.get("valid") != "yes":
                continue
            # results.csv of the first version has no outcome column
            row.setdefault("outcome", "completed" if row.get("stop_reason") == "time" else row.get("stop_reason", ""))
            own_goal = number(models.get(row["model"], {}).get("goal"))
            runs.append(SavedRun(session, run_folder(session, row), row, own_goal,
                                 not_comparable(row, own_goal)))
    return sorted(runs, key=lambda r: r.row.get("time", ""))
