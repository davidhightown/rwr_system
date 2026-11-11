#!/usr/bin/env python3
"""The runs of eval_policies.py on W&B, in the project and format of the other real-hand evaluation
(rwr-skin/real_hand_eval, written by real_eval/upload_wandb.py on the branch feat/real-hand-evaluation), so that the
results of both are in one place:

- one W&B run per run of a policy, job type "real_hand", grouped and named by the model: the name W&B already has for
  it (e.g. "48_ibex"), else the file name without _policy.pt,
- config: model.*, train.* and sim.* (the training run and its sim evaluation, taken over from an earlier W&B run of
  the same model), eval.* (settings and conditions; eval.evaluator = "eval_policies.py"), code.*,
- summary: result/*, cube/*, hand/* (the other evaluation's keys where a figure means the same; note that its runs end
  at the first drop, while here the run continues after a drop: result/turns_before_first_drop is its result/turns),
- curves over policy time (curve/turns, curve/max_temp_c, curve/total_current_a against time_s),
- files: the run folder's CSV files and meta.json; video.mp4 only when asked for (about 25 MB),
- the run "leaderboard": per model and evaluator, its runs at its own goal speed with a cold start, from all runs on W&B.

A model counts as tested when W&B has a real_hand run of the same policy file (SHA-256) or of the same model name.

Needs wandb and a login (`wandb login`). As a script:

    python cube_eval/wandb_results.py untested models/           # the models in models/ without a run on W&B
    python cube_eval/wandb_results.py upload logs/real_eval/20261111_203512_ibex [--video]

The parts that do not talk to W&B (what goes into a run, the leaderboard, which models are tested) are plain functions
on dicts; tests/test_cube_eval.py tests them.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ENTITY = "rwr-skin"
PROJECT = "real_hand_eval"
EVALUATOR = "eval_policies.py"
OTHER_EVALUATOR = "evaluate_real_policies.py"  # runs without eval.evaluator
REFERENCE_MODEL = "48_ibex"  # as in the other evaluation: run it now and then to see whether the hand changed
MAX_START_C = 42.5  # cold start: the hottest motor at most this warm when the run starts (as in the other evaluation)
CURVE_STEP_S = 0.5
FILES = ("meta.json", "turns.csv", "trace.csv", "joints.csv", "joint_tracking.csv", "temperatures.csv", "health.csv")
UPLOADED_OUTCOMES = ("completed", "too many drops")  # the others say nothing about the policy

# results.csv column of eval_policies.py -> summary key
SUMMARY = {
    "turns": "result/turns", "mean_speed_rad_s": "result/rad_per_s", "active_s": "result/duration_s",
    "outcome": "result/outcome", "turns_30s": "result/turns_30s", "best_10s_rad_per_s": "result/best_10s_rad_per_s",
    "longest_stall_s": "result/longest_stall_s", "turns_before_first_drop": "result/turns_before_first_drop",
    "drops": "result/drops", "first_drop_s": "result/first_drop_s", "pause_drop_s": "result/pause_drop_s",
    "wall_s": "result/wall_s",
    "tracked_fraction": "cube/tracked_fraction", "detections_per_s": "cube/poses_per_s",
    "rejected_share": "cube/rejected_poses", "longest_gap_s": "cube/longest_gap_s", "gap_s": "cube/gap_s",
    "tilt_turns": "cube/tilt_turns", "cube_offset_mean_mm": "cube/offset_mean_mm",
    "cube_offset_max_mm": "cube/offset_max_mm", "camera_tilt_deg": "cube/camera_tilt_deg",
    "camera_moved_deg": "cube/camera_moved_deg",
    "temp_max_start_c": "hand/start_temp_max_c", "temp_mean_start_c": "hand/start_temp_mean_c",
    "temp_peak_c": "hand/run_temp_max_c", "temp_max_end_c": "hand/end_temp_max_c", "hottest_joint": "hand/hottest_joint",
    "heating_max_c_per_min": "hand/heating_max_c_per_min", "heating_max_joint": "hand/heating_max_joint",
    "heat_pauses": "hand/heat_pauses", "pause_heat_s": "hand/pause_heat_s",
    "current_total_mean_a": "hand/total_current_a", "highest_current_joint": "hand/highest_current_joint",
    "current_peak_ma": "hand/current_peak_ma", "min_voltage_v": "hand/min_voltage_v",
    "servo_errors": "hand/servo_errors", "temperature_glitches": "hand/temperature_glitches",
    "joint_delay_median_ms": "hand/median_lag_ms", "worst_joint": "hand/worst_joint",
    "worst_joint_error_rms_deg": "hand/worst_joint_error_deg", "joint_error_rms_deg": "hand/joint_error_rms_deg",
}
STOP_REASONS = {"completed": "time", "too many drops": "too many drops"}


def number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_name(policy_path):
    """The model's name from its file, as eval_policies.py names it: ibex_policy.pt -> ibex."""
    return Path(policy_path).stem.removesuffix("_policy")


def retry(call, attempts=4):
    """W&B's API sometimes times out."""
    for i in range(attempts):
        try:
            return call()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(3 * (i + 1))


# -- what W&B knows (records: {"id", "job_type", "config", "summary"} per W&B run)
def known_models(records):
    """{short name: dict(name, shas, runs, idx, description, created, checkpoint_iteration, train, sim, own_goal)} from
    the real_hand runs on W&B. The first record that has a field gives it."""
    out = {}
    for r in records:
        if r.get("job_type") != "real_hand":
            continue
        config = r.get("config") or {}
        m = config.get("model") or {}
        short = m.get("short_name") or m.get("name")
        if not short:
            continue
        k = out.setdefault(short, dict(name=m.get("name") or short, shas=set(), runs=0))
        k["runs"] += 1
        if m.get("policy_sha256"):
            k["shas"].add(m["policy_sha256"])
        for key in ("idx", "description", "created", "checkpoint_iteration"):
            if m.get(key) not in (None, ""):
                k.setdefault(key, m[key])
        if (config.get("train") or {}).get("run"):
            k.setdefault("train", config["train"])
        if config.get("sim"):
            k.setdefault("sim", config["sim"])
        own = (config.get("eval") or {}).get("own_goal_rad_s")
        if own is not None:
            k.setdefault("own_goal", own)
    return out


def find_model(known, name, sha=""):
    """(short name on W&B, its entry) for a model file: the same file, else the same name (any case); (None, None)."""
    if sha:
        for short, k in known.items():
            if sha in k["shas"]:
                return short, k
    for short, k in known.items():
        if short.lower() == name.lower():
            return short, k
    return None, None


def runs(entry):
    return f"{entry['runs']} run" + ("s" if entry["runs"] != 1 else "")


def tested_because(known, name, sha):
    """Why a model counts as tested ("" = it does not)."""
    if sha:
        for short, k in known.items():
            if sha in k["shas"]:
                return f"same file as {k['name']} ({runs(k)} on W&B)"
    short, k = find_model(known, name)
    return f"{k['name']} has {runs(k)} on W&B" if k else ""


def split_models(models_dir, known):
    """The models in models_dir (*_policy.pt, by name): ([untested files], [(tested file, why)])."""
    todo, done = [], []
    for path in sorted(Path(models_dir).glob("*_policy.pt")):
        why = tested_because(known, model_name(path), sha256(path))
        (done.append((path, why)) if why else todo.append(path))
    return todo, done


# -- what goes into a W&B run
def run_id(meta, run_dir):
    """Stable, unique and in the style of the other evaluation's ids."""
    key = f"{meta.get('host', '')}/{Path(run_dir).name}"
    name = re.sub(r"[^a-z0-9_-]", "-", Path(run_dir).name.lower())[:56]
    return f"{name}-{hashlib.sha1(key.encode()).hexdigest()[:6]}"


def config_of(meta, run_dir, known):
    """The W&B config of a run from its meta.json."""
    run_dir = Path(run_dir)
    options = meta.get("options") or {}
    name = meta["model"]
    sha = meta.get("policy_sha256_full") or meta.get("policy_sha256", "")
    short, k = find_model(known, name, sha)
    k = k or {}
    goal = number(meta.get("goal_rad_s"))
    own = number(k.get("own_goal"))
    start = number(meta.get("temp_max_start_c"))
    commit = str(meta.get("git_commit", ""))
    sim = dict(k.get("sim") or {})
    return dict(
        model=dict(name=k.get("name", name), short_name=short or name, idx=k.get("idx"),
                   description=k.get("description", ""), checkpoint_iteration=k.get("checkpoint_iteration"),
                   created=k.get("created", ""), policy_sha256=sha, policy_file=meta.get("policy_file", "")),
        train=dict(k.get("train") or {}),
        sim=sim,
        eval=dict(evaluator=EVALUATOR, session=run_dir.parent.name, run_folder=run_dir.name,
                  started=meta.get("time", ""), date=str(meta.get("time", ""))[:10], note=meta.get("note", ""),
                  host=meta.get("host", ""), goal_rad_s=goal, own_goal_rad_s=own,
                  at_own_goal=own is None or goal is None or abs(goal - own) <= 1e-6,
                  start_temp_max_c=start, cold_start=start is None or start <= MAX_START_C,
                  reference=k.get("name", name) == REFERENCE_MODEL,
                  duration_s=number(options.get("duration")), temp_limit_c=number(options.get("temp_limit")),
                  temp_margin_c=number(options.get("temp_margin")),
                  start_wrist_deg=number(options.get("start_wrist_deg")), drop_dist_m=number(options.get("drop_dist")),
                  lost_timeout_s=number(options.get("lost_timeout")), max_drops=number(options.get("max_drops")),
                  exposure_us=number(options.get("exposure_us")), camera=(meta.get("camera") or {}).get("camera_id")),
        code=dict(commit=commit.removesuffix("+changes"), uncommitted_changes=commit.endswith("+changes"),
                  hand_config_sha256=meta.get("hand_config_sha256", "")))


def summary_of(meta):
    out = {}
    for column, key in SUMMARY.items():
        value = meta.get(column, "")
        if value not in ("", None) or column == "servo_errors":  # an empty figure is left out (a W&B key keeps
            # one type); no servo errors stays "", as in the other evaluation
            x = number(value)
            out[key] = x if x is not None else value
    out["cube/turns_camera_axis"] = out.get("result/turns")  # turns are counted about the camera's viewing axis
    out["result/stop_reason"] = STOP_REASONS.get(meta.get("outcome", ""), meta.get("outcome", ""))
    drops = number(meta.get("drops")) or 0
    out["result/dropped"] = drops > 0
    out["result/full_60s"] = meta.get("outcome") == "completed" and drops == 0  # held the cube the whole time
    return out


def tags_of(config, summary):
    return [t for t in (summary.get("result/outcome", ""), "reference" if config["eval"]["reference"] else "",
                        EVALUATOR) if t]


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def curves(run_dir, duration):
    """Every CURVE_STEP_S of policy time: the turns, the hottest motor and the total |current| of all motors."""
    run_dir = Path(run_dir)
    if not duration or duration <= 0:
        return []
    grid = np.append(np.arange(0.0, duration, CURVE_STEP_S), duration)
    points = [{"time_s": round(float(g), 3)} for g in grid]
    turns = [(number(r["policy_time_s"]), number(r["turns"])) for r in read_csv(run_dir / "turns.csv")]
    turns = [(t, c) for t, c in turns if t is not None and c is not None]
    t = np.array([0.0] + [t for t, _ in turns])
    c = np.array([0.0] + [c for _, c in turns])
    for p in points:  # the last counted value before each point
        p["curve/turns"] = float(c[np.searchsorted(t, p["time_s"], side="right") - 1])
    temps = [r for r in read_csv(run_dir / "temperatures.csv") if number(r.get("policy_time_s")) is not None
             and r.get("state") == "running"]
    if temps:
        joints = [k for k in temps[0] if k not in ("time_s", "policy_time_s", "state")]
        rows = [(number(r["policy_time_s"]), max([number(r[j]) for j in joints if number(r[j]) is not None],
                                                 default=None)) for r in temps]
        rows = [(a, b) for a, b in rows if b is not None]
        if rows:
            tt, hot = np.array(rows).T
            for p in points:
                p["curve/max_temp_c"] = float(np.interp(p["time_s"], tt, hot))
    joints_csv = run_dir / "joints.csv"
    if joints_csv.exists():
        with open(joints_csv, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            cols = [i for i, h in enumerate(header) if h.startswith("current_ma_")]
            data = [(float(r[0]), sum(abs(float(r[i])) for i in cols)) for r in reader if cols and r[cols[0]] != ""]
        if data:
            jt, cur = np.array(data).T
            for p in points:
                window = (jt >= p["time_s"] - CURVE_STEP_S / 2) & (jt < p["time_s"] + CURVE_STEP_S / 2)
                if window.any():
                    p["curve/total_current_a"] = float(cur[window].mean() / 1000.0)
    return points


def leaderboard_rows(records):
    """As the other evaluation's leaderboard, from the runs on W&B, with one row per model and evaluator (the two count
    differently: there a run ends at the first drop), the best mean first."""
    groups = {}
    for r in records:
        if r.get("job_type") != "real_hand":
            continue
        config, s = r.get("config") or {}, r.get("summary") or {}
        e, m = config.get("eval") or {}, config.get("model") or {}
        if not (e.get("at_own_goal", True) and e.get("cold_start", True)) or number(s.get("result/turns")) is None:
            continue
        key = (m.get("name") or m.get("short_name") or "?", e.get("evaluator") or OTHER_EVALUATOR)
        groups.setdefault(key, []).append(r)
    rows = []
    for (name, evaluator), rs in groups.items():
        rs = sorted(rs, key=lambda r: str((r["config"].get("eval") or {}).get("started", "")))
        turns = [number(r["summary"]["result/turns"]) for r in rs]
        first = rs[0]["config"]
        rows.append(dict(model=name, description=(first.get("model") or {}).get("description", ""), runs=len(rs),
                         mean_turns=round(statistics.mean(turns), 2),
                         std_turns=round(statistics.stdev(turns), 2) if len(turns) > 1 else None,
                         best_turns=round(max(turns), 2), worst_turns=round(min(turns), 2),
                         full_60s_runs=sum(bool(r["summary"].get("result/full_60s")) for r in rs),
                         mean_duration_s=round(statistics.mean(number(r["summary"].get("result/duration_s")) or 0.0
                                                               for r in rs), 1),
                         each_run=" ".join(f"{t:.1f}" for t in turns),
                         training_url=(first.get("train") or {}).get("url", ""), evaluator=evaluator))
    return sorted(rows, key=lambda row: -row["mean_turns"])


# -- talking to W&B
class Project:
    def __init__(self, entity=ENTITY, project=PROJECT):
        import wandb
        self.wandb, self.entity, self.project = wandb, entity, project
        self.api = wandb.Api(timeout=60)

    @property
    def path(self):
        return f"{self.entity}/{self.project}"

    def records(self):
        try:
            runs = retry(lambda: list(self.api.runs(self.path)))
        except (ValueError, self.wandb.errors.CommError):  # the project does not exist yet
            return []
        out = []
        for r in runs:
            keys = [k for k in r.summary.keys() if k.startswith(("result/", "cube/", "hand/"))]
            out.append(dict(id=r.id, job_type=r.job_type, config=dict(r.config),
                            summary={k: r.summary.get(k) for k in keys}))
        return out

    def upload(self, run_dir, video=False, records=None, wandb_dir=None):
        """One run folder of eval_policies.py; returns (its url, the new record), or (None, None) if it is on W&B."""
        wandb = self.wandb
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text())
        records = self.records() if records is None else records
        rid = run_id(meta, run_dir)
        if any(r["id"] == rid for r in records):
            return None, None
        config = config_of(meta, run_dir, known_models(records))
        summary = summary_of(meta)
        run = wandb.init(entity=self.entity, project=self.project, id=rid,
                         name=f"{config['model']['name']} {str(meta.get('time', ''))[:16]}".strip(),
                         group=config["model"]["name"], job_type="real_hand", tags=tags_of(config, summary),
                         config=config, reinit="finish_previous", dir=str(wandb_dir or run_dir.parent),
                         settings=wandb.Settings(silent=True))
        run.define_metric("time_s")
        run.define_metric("curve/*", step_metric="time_s")
        for point in curves(run_dir, number(meta.get("active_s"))):
            run.log(point)
        run.summary.update(summary)
        for name in FILES + (("video.mp4",) if video else ()):
            if (run_dir / name).exists():
                run.save(str(run_dir / name), base_path=str(run_dir), policy="now")
        url = run.url
        run.finish()
        return url, dict(id=rid, job_type="real_hand", config=config, summary=summary)

    def write_leaderboard(self, records, wandb_dir):
        rows = leaderboard_rows(records)
        if not rows:
            return
        wandb = self.wandb
        board = wandb.init(entity=self.entity, project=self.project, id="leaderboard", name="leaderboard",
                           group="leaderboard", job_type="leaderboard", resume="allow", reinit="finish_previous",
                           dir=str(wandb_dir), settings=wandb.Settings(silent=True))
        board.log({"leaderboard": wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])})
        board.finish()


def upload_runs(run_dirs, video=False, entity=ENTITY, project=PROJECT):
    """Upload run folders (skipping those already on W&B), then rewrite the leaderboard. One upload at a time: a second
    one waits for the first (the leaderboard of the later one then has both)."""
    run_dirs = [Path(d) for d in run_dirs]
    lock_dir = run_dirs[0].parent / "wandb"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = open(lock_dir / "upload.lock", "w")  # released when the process ends
    fcntl.flock(lock, fcntl.LOCK_EX)
    p = Project(entity, project)
    records = p.records()
    new = 0
    for d in run_dirs:
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("outcome") not in UPLOADED_OUTCOMES:
            print(f"  {d.name}: not uploaded ({meta.get('outcome')})")
            continue
        url, record = p.upload(d, video=video, records=records, wandb_dir=d.parent)
        if record is None:
            print(f"  {d.name}: already on W&B")
            continue
        records.append(record)
        new += 1
        print(f"  {d.name}: {url or 'uploaded'}")
    if new:
        p.write_leaderboard(records, run_dirs[0].parent)
        print(f"  leaderboard written: https://wandb.ai/{p.path}")
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--project", default=PROJECT)
    sub = ap.add_subparsers(dest="command", required=True)
    u = sub.add_parser("untested", help="List the models in a folder without a run on W&B.")
    u.add_argument("models_dir", type=Path)
    up = sub.add_parser("upload", help="Upload run folders of eval_policies.py.")
    up.add_argument("run_dirs", nargs="+", type=Path)
    up.add_argument("--video", action="store_true", help="Also video.mp4.")
    args = ap.parse_args(argv)
    if args.command == "untested":
        todo, done = split_models(args.models_dir, known_models(Project(args.entity, args.project).records()))
        for path, why in done:
            print(f"  tested    {path.name:28s} {why}")
        for path in todo:
            print(f"  untested  {path.name}")
        return 0
    upload_runs(args.run_dirs, video=args.video, entity=args.entity, project=args.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
