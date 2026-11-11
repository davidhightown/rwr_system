"""Upload the saved real-hand runs of evaluate_real_policies.py to W&B: one W&B run per saved run, grouped by model.

Each W&B run has
- config: the model (with its description from models.json), its training run and the standard sim evaluation of
  that run (sim.*), the evaluation settings and conditions (eval.*: goal speed and whether it is the model's own,
  start temperature, cold start), the code version (code.*),
- summary: the results (result/*), the cube tracking (cube/*) and the hand (hand/*),
- curves over the run's time: turns, the hottest motor and the total motor current (curve/*), and the two plots,
- files: meta.json, trace.csv, joints.csv, policy_log.csv, policy_node.log, temperatures.csv, motor_status.csv,
  joint_tracking.csv and the session's code changes; video.mp4 only for the runs named with --video.

A separate run "leaderboard" holds one table of the runs on W&B: per model, the runs at its own goal speed with a
cold start (mean, spread, best, worst); the rules are in results_rules.py. Runs already on W&B are skipped, so this
can run any time; evaluate_real_policies.py --wandb runs it with --session after every saved run.
Needs wandb and numpy (e.g. the Isaac Lab env, not the ROS workspace):

    python upload_wandb.py --dry-run          # list what would be uploaded
    python upload_wandb.py                    # upload the new saved runs of all sessions and the leaderboard
    python upload_wandb.py --session eval_20261110_173525   # only the runs of this session (and the leaderboard)
    python upload_wandb.py --update           # also rewrite config, tags and summary of the runs already on W&B
                                              # (after a change of models.json or of the rules; W&B merges the
                                              # config, so a key the script no longer writes stays on W&B)
    python upload_wandb.py --video 48_ibex_212216 37_mild_critic_v2_210734   # add these runs' videos
"""

import argparse
import csv
import fcntl
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from results_rules import REFERENCE_MODEL, number, load_models, saved_runs

FILES = ("meta.json", "trace.csv", "joints.csv", "policy_log.csv", "policy_node.log", "temperatures.csv",
         "motor_status.csv", "joint_tracking.csv")
PLOTS = ("turns_vs_time", "joints")
CURVE_STEP_S = 0.5
SIM_KEYS = ("turns", "drop_rate", "speed_along_cmd", "speed_error", "position_error", "action_rate",
            "in_hand_fraction", "time_alive_s", "episode_s", "checkpoint")  # eval/* of the training run
# results.csv column -> summary key
SUMMARY = {
    "turns": "result/turns", "speed_along_cmd": "result/rad_per_s", "duration_s": "result/duration_s",
    "outcome": "result/outcome", "stop_reason": "result/stop_reason", "turns_30s": "result/turns_30s",
    "best_10s_rad_per_s": "result/best_10s_rad_per_s", "longest_stall_s": "result/longest_stall_s",
    "tracked_fraction": "cube/tracked_fraction", "poses_per_s": "cube/poses_per_s",
    "rejected_poses": "cube/rejected_poses", "longest_gap_s": "cube/longest_gap_s",
    "turns_camera_axis": "cube/turns_camera_axis", "spin_axis_tilt_deg": "cube/spin_axis_tilt_deg",
    "warnings": "cube/warnings",
    "start_temp_max": "hand/start_temp_max_c", "run_temp_max": "hand/run_temp_max_c",
    "hottest_joint": "hand/hottest_joint",
    "total_current_a": "hand/total_current_a", "highest_current_joint": "hand/highest_current_joint",
    "min_voltage_v": "hand/min_voltage_v", "servo_errors": "hand/servo_errors",
    "servo_errors_parking": "hand/servo_errors_parking", "temperature_glitches": "hand/temperature_glitches",
    "median_lag_ms": "hand/median_lag_ms", "worst_joint": "hand/worst_joint",
    "worst_joint_error_deg": "hand/worst_joint_error_deg", "stuck_joints": "hand/stuck_joints",
}


def retry(call, attempts=4):
    """W&B's API sometimes times out ("the service process is busy")."""
    for i in range(attempts):
        try:
            return call()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(3 * (i + 1))


def read_meta(folder: Path) -> dict:
    path = folder / "meta.json"
    return json.load(open(path)) if path.exists() else {}


def code_changes_file(meta: dict) -> str:
    """The session's code_changes_<time>.diff that was in effect for the run ("" without uncommitted changes)."""
    return meta.get("session", {}).get("rwr_system", {}).get("uncommitted_changes_file", "") or ""


class TrainingRuns:
    """The training runs on W&B, each fetched once."""

    def __init__(self, api):
        self.api, self.cache = api, {}

    def sim(self, path: str) -> dict:
        """The standard sim evaluation (eval/*) of a training run, {} without one."""
        if not path:
            return {}
        if path not in self.cache:
            summary = retry(lambda: self.api.run(path)).summary
            self.cache[path] = {k: summary.get(f"eval/{k}") for k in SIM_KEYS if summary.get(f"eval/{k}") is not None}
        return self.cache[path]


# -- what goes into a W&B run
def config_of(run, model: dict, meta: dict, sim: dict) -> dict:
    session = meta.get("session", {})
    options, code = session.get("options", {}), session.get("rwr_system", {})
    training = model.get("wandb", {})
    sim = dict(sim)
    if "checkpoint" in sim:  # the sim evaluation may be of another checkpoint than the one tested here
        sim["same_checkpoint"] = Path(str(sim["checkpoint"])).name == model.get("ckpt")  # e.g. model_4998.pt
    return dict(
        model=dict(name=run.model, short_name=model.get("name", run.model[3:]), idx=model.get("idx"),
                   description=model.get("description", ""), checkpoint_iteration=model.get("it"),
                   created=model.get("created", ""), policy_sha256=meta.get("model", {}).get("policy_sha256", "")),
        train=dict(run=training.get("path", ""), url=training.get("url", ""), run_name=training.get("run_name", "")),
        sim=sim,
        eval=dict(session=run.session.name, run_folder=run.folder.name, started=run.row.get("time", ""),
                  note=run.row.get("note", ""), goal_rad_s=number(run.row.get("goal_ang_vel")),
                  own_goal_rad_s=run.own_goal, at_own_goal=run.at_own_goal, start_temp_max_c=run.start_temp,
                  cold_start=run.cold_start, date=run.row.get("time", "")[:10], reference=run.model == REFERENCE_MODEL,
                  duration_s=options.get("duration"), start_temp_c=options.get("start_temp"),
                  abort_temp_c=options.get("abort_temp"), start_wrist_deg=options.get("start_wrist_deg"),
                  drop_dist_m=options.get("drop_dist"), lost_timeout_s=options.get("lost_timeout"),
                  exposure_us=options.get("exposure_us")),
        code=dict(commit=code.get("commit", ""), uncommitted_changes=code_changes_file(meta),
                  hand_config_sha256=session.get("hand_config", {}).get("sha256", "")))


def summary_of(run) -> dict:
    out = {}
    for column, key in SUMMARY.items():
        value = run.row.get(column, "")
        x = number(value)
        out[key] = x if x is not None else value
    out["result/dropped"] = run.row.get("dropped") == "True"
    out["result/full_60s"] = run.row.get("outcome") == "completed"
    return out


def display_name(run) -> str:
    """e.g. "48_ibex 2026-11-10 21:22": the model and when the run started."""
    return f"{run.model} {run.row.get('time', '')[:16]}".strip()


def tags_of(run) -> list[str]:
    return [t for t in (run.row.get("outcome", ""), "reference" if run.model == REFERENCE_MODEL else "") if t]


def curves(folder: Path, goal: float, axis_sign: float = -1.0) -> list[dict]:
    """Every CURVE_STEP_S from the run's start: the turns as the evaluation counted them (about the spin axis it
    estimated from all accepted rotation steps; axis_sign: --axis-sign of the run, the camera's up axis), the hottest
    motor and the total |current| of all motors."""
    trace = folder / "trace.csv"
    if not trace.exists():
        return []
    rows = list(csv.DictReader(open(trace)))
    if not rows:
        return []
    t0 = float(rows[0]["time"])  # the turn counter starts with the run
    acc = [r for r in rows if r["accepted"] == "1"]
    t = np.array([float(r["time"]) - t0 for r in acc])
    steps = np.array([[float(r["step_rx"]), float(r["step_ry"]), float(r["step_rz"])] for r in acc]).reshape(-1, 3)
    up = np.array([0.0, 0.0, axis_sign])
    total = steps.sum(axis=0)
    axis = up
    if np.linalg.norm(total) > np.radians(90):  # as TurnCounter.result()
        axis = total / np.linalg.norm(total)
        axis = axis if axis @ up >= 0 else -axis
    turns = (1.0 if goal >= 0 else -1.0) * np.cumsum(steps @ axis) / (2 * np.pi)
    end = max(t[-1] if len(t) else 0.0, float(rows[-1]["time"]) - t0)
    grid = np.arange(0.0, end, CURVE_STEP_S)
    grid = np.append(grid, end)  # the last point at the last detection: the curve ends at the run's turns
    points = [{"time_s": float(g), "curve/turns": float(np.interp(g, t, turns)) if len(t) else 0.0} for g in grid]

    temps = folder / "temperatures.csv"
    if temps.exists():
        rows_t = [[float(x) for x in r] for r in list(csv.reader(open(temps)))[1:] if r]
        if rows_t:
            tt = np.array([r[0] for r in rows_t])
            hottest = np.array([max(r[1:]) for r in rows_t])
            for p in points:
                p["curve/max_temp_c"] = float(np.interp(p["time_s"], tt, hottest))
    joints = folder / "joints.csv"
    if joints.exists():
        with open(joints) as f:
            reader = csv.reader(f)
            header = next(reader)
            cols = [i for i, h in enumerate(header) if h.startswith("current_ma_")]
            data = [(float(r[0]), sum(abs(float(r[i])) for i in cols)) for r in reader if cols and r[cols[0]] != ""]
        if data:
            jt, cur = np.array(data).T
            for p in points:
                window = (jt >= p["time_s"] - CURVE_STEP_S / 2) & (jt < p["time_s"] + CURVE_STEP_S / 2)
                if window.any():
                    p["curve/total_current_a"] = float(cur[window].mean() / 1000.0)
    return points


def leaderboard(runs, models) -> list[dict]:
    """Per model, its runs at its own goal speed with a cold start; the best mean first."""
    by_model = {}
    for r in runs:
        if r.comparable:
            by_model.setdefault(r.model, []).append(r)
    rows = []
    for name, rs in by_model.items():
        turns = [number(r.row["turns"]) or 0.0 for r in rs]
        m = models.get(name, {})
        rows.append(dict(model=name, description=m.get("description", ""), runs=len(rs),
                         mean_turns=round(statistics.mean(turns), 2),
                         std_turns=round(statistics.stdev(turns), 2) if len(turns) > 1 else None,
                         best_turns=round(max(turns), 2), worst_turns=round(min(turns), 2),
                         full_60s_runs=sum(r.row.get("outcome") == "completed" for r in rs),
                         mean_duration_s=round(statistics.mean(number(r.row["duration_s"]) or 0.0 for r in rs), 1),
                         each_run=" ".join(f"{t:.1f}" for t in turns), training_url=m.get("wandb", {}).get("url", "")))
    return sorted(rows, key=lambda row: -row["mean_turns"])


# -- upload
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", default=os.path.expanduser("~/policies/real_test"))
    p.add_argument("--entity", default="rwr-skin")
    p.add_argument("--project", default="real_hand_eval")
    p.add_argument("--update", action="store_true", help="Also rewrite config, tags and summary of uploaded runs.")
    p.add_argument("--video", nargs="+", default=[], metavar="RUN_FOLDER",
                   help="Add video.mp4 of these runs (run folder names, e.g. 48_ibex_212216; about 25 MB each).")
    p.add_argument("--session", nargs="+", default=[], metavar="SESSION_FOLDER",
                   help="Only the runs of these sessions (e.g. eval_20261110_173525, also in archive/).")
    p.add_argument("--only", nargs="+", default=[], metavar="RUN_FOLDER",
                   help="Only these runs (run folder names); the leaderboard is not written. For a trial upload.")
    p.add_argument("--dry-run", action="store_true", help="Only list what would be uploaded.")
    args = p.parse_args()

    import wandb  # here: --help works without it

    models_dir = Path(args.models_dir)
    # one upload at a time (evaluate_real_policies.py starts one after every saved run): a second one waits here,
    # then sees the runs the first one uploaded
    (models_dir / "wandb").mkdir(parents=True, exist_ok=True)
    lock = open(models_dir / "wandb" / "upload.lock", "w")  # released when this process ends
    fcntl.flock(lock, fcntl.LOCK_EX)
    models = load_models(models_dir)
    runs = saved_runs(models_dir)
    unknown = set(args.video + args.only) - {r.folder.name for r in runs}
    if unknown:
        raise SystemExit(f"not saved runs (run folder names): {sorted(unknown)}")
    unknown_sessions = set(args.session) - {r.session.name for r in runs}
    if unknown_sessions:
        raise SystemExit(f"no saved runs in these sessions: {sorted(unknown_sessions)}")
    selected = [r for r in runs if (not args.only or r.folder.name in args.only)
                and (not args.session or r.session.name in args.session)]

    api = wandb.Api(timeout=60)
    project = f"{args.entity}/{args.project}"
    try:
        uploaded = {r.id for r in retry(lambda: list(api.runs(project)))}
    except (ValueError, wandb.errors.CommError):  # the project does not exist yet
        uploaded = set()
    new = [r for r in selected if r.run_id not in uploaded]
    existing = [r for r in selected if r.run_id in uploaded]
    print(f"{len(selected)} saved runs: {len(new)} new, {len(existing)} already in {project}"
          + (" (config, tags and summary are rewritten)" if args.update and existing else ""))
    for r in new:
        print(f"  new  {r.folder.name:34s} {r.model:26s} {number(r.row.get('turns')) or 0:5.1f} turns  "
              f"{'own goal, cold start' if r.comparable else r.not_comparable_because}")
    not_uploaded = [name for name in args.video
                    if all(r.folder.name != name or (r.run_id not in uploaded and r not in new) for r in runs)]
    if not_uploaded:
        raise SystemExit(f"upload these runs before their videos (without --only): {not_uploaded}")
    if args.video:
        print(f"  videos: {', '.join(args.video)}")
    if args.dry_run:
        return

    training = TrainingRuns(api)
    for r in new:
        model = models.get(r.model, {})
        meta = read_meta(r.folder)
        run = wandb.init(entity=args.entity, project=args.project, id=r.run_id, name=display_name(r), group=r.model,
                         job_type="real_hand", tags=tags_of(r),
                         config=config_of(r, model, meta, training.sim(model.get("wandb", {}).get("path", ""))),
                         reinit="finish_previous", dir=str(models_dir), settings=wandb.Settings(silent=True))
        run.define_metric("time_s")
        run.define_metric("curve/*", step_metric="time_s")
        axis_sign = float(meta.get("session", {}).get("options", {}).get("axis_sign", -1.0))
        for point in curves(r.folder, number(r.row.get("goal_ang_vel")) or 1.0, axis_sign):
            run.log(point)
        run.log({f"plots/{name}": wandb.Image(str(r.folder / f"{name}.png"))
                 for name in PLOTS if (r.folder / f"{name}.png").exists()})
        run.summary.update(summary_of(r))
        for f in FILES:
            if (r.folder / f).exists():
                run.save(str(r.folder / f), base_path=str(r.folder), policy="now")
        diff = code_changes_file(meta)
        if diff and (r.session / diff).exists():
            run.save(str(r.session / diff), base_path=str(r.session), policy="now")
        run.finish()
        print(f"  uploaded {r.folder.name}")

    if args.update:
        for r in existing:
            model = models.get(r.model, {})
            remote = retry(lambda: api.run(f"{project}/{r.run_id}"))
            sim = training.sim(model.get("wandb", {}).get("path", ""))
            remote.config.update(config_of(r, model, read_meta(r.folder), sim))
            remote.tags = tags_of(r)
            remote.group = r.model
            remote.name = display_name(r)
            remote.summary.update(summary_of(r))
            retry(remote.update)
        print(f"  updated {len(existing)} runs")

    for name in args.video:
        r = next(r for r in runs if r.folder.name == name)
        video = r.folder / "video.mp4"
        if not video.exists():
            print(f"  {name}: no video.mp4")
            continue
        run = wandb.init(entity=args.entity, project=args.project, id=r.run_id, resume="must",
                         reinit="finish_previous", dir=str(models_dir), settings=wandb.Settings(silent=True))
        run.save(str(video), base_path=str(r.folder), policy="now")
        run.finish()
        print(f"  video of {name} uploaded")

    if not args.only and (new or args.update):
        board = wandb.init(entity=args.entity, project=args.project, id="leaderboard", name="leaderboard",
                           group="leaderboard", job_type="leaderboard", resume="allow", reinit="finish_previous",
                           dir=str(models_dir), settings=wandb.Settings(silent=True))
        on_wandb = uploaded | {r.run_id for r in new}  # not the sessions that were never uploaded
        rows = leaderboard([r for r in runs if r.run_id in on_wandb], models)
        board.log({"leaderboard": wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])})
        board.finish()
        print("  leaderboard written")
    print(f"https://wandb.ai/{project}")


if __name__ == "__main__":
    main()
