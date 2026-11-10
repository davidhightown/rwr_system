# ORCA v2 chirp recording

Record joint chirps on the ORCA v2 hand through `orca_core`, without ROS, to identify the servo model.

## Which script?

| Script | What it is | Use it for |
|---|---|---|
| `chirp.py` | The original recorder. It sweeps the joints you list (default: only `index_mcp`) and writes the CSVs. Nothing else. | One-off checks, custom sweeps |
| `chirp_all.py` | The newer wrapper around `chirp.py`. It has the same sweep, plus SysID defaults, checks before and after the run, and a tagged run folder (see below). | **Every SysID recording**, including a single finger |

`chirp_all.py` accepts every `chirp.py` flag and changes these defaults:

| | `chirp.py` | `chirp_all.py` |
|---|---|---|
| `--config` | the repo's `src/.../orcahand_v2/right/config.yaml` | the installed `~/ros_ws/install/.../orcahand_v2/right/config.yaml` |
| `--joints` | `index_mcp` | all 17, with the `--finger` joints first |
| `--steps-deg` | none | `2 6 12` (steps after each chirp) |
| `--abd-clear-all` | off | **always on** (other fingers go to their abduction limits) |

`chirp_all.py` also:
- runs from a copy of `config.yaml` and `calibration.yaml`, so a symlinked install config works;
- warns before moving if a joint's calibrated motor travel looks stalled;
- prints a check table and opens `overview.png` after the run;
- saves the config and calibration in the run folder and tags the folder `_good`/`_bad`/...

Both scripts only print the plan by default. **Nothing moves without `--execute`.**

## Setup

1. **Python.** On macOS, use the `rwr_jazzy` environment (see the [main README](../../README.md)). It already contains
   `orca_core` and `scipy`. Otherwise, create a separate environment from the repo root. `orca_core`'s NumPy
   conflicts with `faive_system`'s pin:
   ```bash
   python3.12 -m venv .venv.chirp && source .venv.chirp/bin/activate
   python -m pip install orca_core==0.5.2 scipy matplotlib
   ```
2. **Calibrate** the hand with the normal v2 workflow ([`src/hand_control/README.md`](../../src/hand_control/README.md)).
   The scripts refuse to run without a complete `calibration.yaml` beside `config.yaml`. They never calibrate on their own.
3. **Connect** power and USB. `port: auto` in `config.yaml` usually works (see Troubleshooting).
4. **Free the port and the hand.** Stop the ROS controller, GUI, tensioning tool and anything else that uses the
   serial port. Leave the hand unloaded, with room to move.

`chirp_all.py` reads the installed config by default (after `colcon build`). For another workspace, pass `--config`.
`chirp.py` needs `--config` pointing at a real folder. It looks for `calibration.yaml` beside the *resolved* path,
and the installed `config.yaml` is a symlink into the source folder. Copy both files out (again after each
recalibration):

```bash
mkdir -p ~/ros_ws/chirp_config
cp -L ~/ros_ws/install/hand_control/share/hand_control/orca/configs/orcahand_v2/right/{config,calibration}.yaml ~/ros_ws/chirp_config/
```

## Run it

```bash
python experiments/orca/chirp_all.py                     # plan only: centers, neighbour poses, duration
python experiments/orca/chirp_all.py --execute           # all 17 joints, about 8 min
```

**First run on a new hand:** check one joint with a small amplitude:

```bash
python experiments/orca/chirp_all.py --joints index_mcp --amplitude-deg 3 --steps-deg --execute
```

Ctrl+C stops the run at any time. The partial data is kept, and the folder is tagged `_interrupted`.

### Only one finger

Pass that finger's three joints to `--joints`, and the same finger to `--finger`:

```bash
python experiments/orca/chirp_all.py --joints ring_abd ring_mcp ring_pip --finger ring            # plan
python experiments/orca/chirp_all.py --joints ring_abd ring_mcp ring_pip --finger ring --execute  # ~1.5 min
```

`--finger` does not select joints. It only records which finger is meant for the fit (`fit_finger` and
`fit_joints` in `metadata.json`), and it defaults to `middle`. If you leave it out, a ring-only run has an empty
`fit_joints`. To include the joints that the minimal SysID set shares, add them too:

```bash
python experiments/orca/chirp_all.py --finger index --execute \
  --joints wrist thumb_cmc thumb_abd thumb_mcp thumb_dip index_abd index_mcp index_pip
```

This is the minimal set (eight joints) for whole-hand SysID. faive_lab reuses that finger's fit for the other
three fingers. This assumes they behave alike, so a full 17-joint run is still the better check.

Joint names: `wrist`, `thumb_cmc`, `thumb_abd`, `thumb_mcp`, `thumb_dip`, and `<finger>_abd`, `<finger>_mcp`,
`<finger>_pip` for `index`, `middle`, `ring` and `pinky`.

### Keeping the other fingers out of the way

Only **finger abduction sweeps** (`index_abd` ... `pinky_abd`) move other joints. With every other finger at
neutral, an abduction sweep runs into the next finger. On the 2026-11 runs, ring_abd stopped about 6° short on the
middle-finger side, and middle_abd on both sides. So, before the sweep, the other fingers' abduction moves away
from the tested finger. They ramp there with the tested joint, hold during the chirp and the steps, and go back
to neutral for the next joint. There are two modes:

| Mode | How | What the other fingers do |
|---|---|---|
| **Clear all** | `--abd-clear-all`. Always on in `chirp_all.py` | Every other finger goes to its abduction ROM limit (minus 2°) on the side away from the tested finger. The sweep stays at the ROM midpoint. Maximum clearance. |
| **Minimum gap** | `chirp.py` default, `--abd-gap-deg 8` | Each finger moves only as far as needed to stay `--abd-gap-deg` from the next finger inward at the sweep's extremes. If a neighbour would leave its ROM, the sweep center moves toward the free side (pinky_abd sweeps at about +13°). |
| **None** | `chirp.py --abd-gap-deg 0` | All other fingers stay at neutral (the old behaviour). |

For example, `ring_abd` in clear-all mode puts index at −23°, middle at −25° and pinky at +28°.
Always look at `neighbour_pose` and `centers_deg` in the printed plan before `--execute`. On a new hand,
check them once with a small amplitude. Both modes assume that positive abduction points toward the pinky, and
the script checks this against the neutral pose.

In `chirp_all.py`, `--abd-gap-deg` has no effect on finger abduction: clear-all is fixed and has no switch to turn
it off. To use the minimum gap or neutral neighbours, run `chirp.py` with the same joints.

Flexion (`_mcp`, `_pip`), thumb and wrist sweeps keep every other joint at neutral. Nothing moves out of the way
for them. When a PIP stops short on both sides (as middle/ring/pinky do on the 2026-11 runs), that points at the
joint's own load (the motor stalls at its current limit), not at a neighbour.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--execute` | off | Connect and move the hand. Without it, only print the plan (`--dry-run` says so explicitly). |
| `--config PATH` | see table above | This hand's `config.yaml`. `calibration.yaml` must be beside it. |
| `--joints J ...` | see table above | Joints to sweep, one after another, in this order. |
| `--finger F` | `middle` | *chirp_all only.* The finger meant for the fit, saved in `metadata.json`. Does not select joints. |
| `--notes TEXT` | asked after the run | *chirp_all only.* Saved in `metadata.json`, e.g. `"re-tensioned index"`. |
| `--amplitude-deg` | 10 | Chirp amplitude about the center. |
| `--center-deg` | ROM midpoint | Sweep center for every joint. center ± max(amplitude, steps) must fit within the ROM. |
| `--f0`, `--f1` | 0.1, 5 | Start and end frequency of the logarithmic sweep (Hz). `f1` must be below `ctrl-freq / 2`. |
| `--duration` | 20 | Seconds per chirp. |
| `--ctrl-freq` | 20 | Command rate (Hz), as in deployment. One row of the chirp CSV per command. |
| `--read-hz` | 100 | Read rate between commands (the fast CSV). A whole multiple of `--ctrl-freq`. 100 Hz loads the bus as the deployment controller does. |
| `--steps-deg S ...` | see table above | After each chirp: steps of +S, back, −S, back about the center. `--steps-deg` with no values turns them off. |
| `--step-hold` | 0.6 | Seconds to hold each step target. |
| `--abd-clear-all` | see table above | Move all other fingers' abduction to their ROM limits during an abduction sweep. |
| `--abd-gap-deg` | 8 | Minimum-gap mode (see above); 0 = neighbours stay at neutral. Ignored for finger abduction when clear-all is on. |
| `--settle-time` | 1.0 | Seconds to wait at the first chirp sample before recording. |
| `--no-ramp` | off | Jump to each joint's start pose at full motor speed instead of ramping over about 1 s. The neighbours also jump. The initial move to neutral still ramps. Recorded as `ramp_to_start`. |
| `--output-dir` | `logs/chirp_v2` | Where run folders go. |

A full `chirp_all.py` run takes about (duration + settle + 1 s ramp + 7.8 s of steps) ≈ 30 s per joint.
The plan prints the estimate.

## Checking a run (`chirp_all.py`)

After the run, the script prints a table per joint (samples, moved = measured/commanded peak-to-peak, mean period,
calibrated motor travel), opens `overview.png`, and asks whether the run was good. The flags in the table:

| Flag | Meaning |
|---|---|
| `low motion` | Moved less than 75% of the commanded peak-to-peak range |
| `stops short -` / `+` | Below 1 Hz, more than 1.5° short of the command on that side. The servo itself loses well under a degree there, so something stops it. On one side only: probably a neighbour. On both: the joint's own load. The motor current in `fast_*.csv` shows which. |
| `short travel` | The calibrated motor travel is more than 25% below the median of the same joint on the other fingers (probably stalled in calibration) |
| `too few samples`, `timing`, `no data` | The recording is incomplete or the control period drifted by more than 5% |

Answer `n` only if the run itself was disturbed (bumped, loaded, loose cable). A genuine outlier is still a good run.
The answer is added to the folder name:

| Suffix | Meaning |
|---|---|
| `_good` | You answered `y` |
| `_bad` | You answered `n` |
| `_untagged` | You pressed Enter or Ctrl+C at the prompt; rename the folder when you decide |
| `_interrupted` | You stopped the run with Ctrl+C; the partial data is kept |
| `_failed` | The run stopped with an error; the partial data is kept |

`chirp.py` does none of this: its folder has only the timestamp, and you check the data with `plot.py`.

## Output

```text
logs/chirp_v2/<UTC timestamp>[_tag]/
├── metadata.json         # plan, centers, neighbour_pose, servo_profile, servo_gains, temperatures, status
├── chirp_<joint>_*.csv   # the read right after each 20 Hz command: timestamp, commanded_position, measured_position
├── fast_<joint>_*.csv    # every read at --read-hz, plus motor_velocity (rad/s) and motor_current (mA)
├── steps_<joint>_*.csv   # the steps, same columns as fast_ (if --steps-deg)
├── config.yaml           # chirp_all only
├── calibration.yaml      # chirp_all only
└── overview.png          # chirp_all only
```

Times are in seconds, angles in degrees, and there are 400 rows per chirp with the defaults. Keep the filenames and
`metadata.json` together. Use only runs with `"status": "complete"` for SysID. On motor-only hands, measured angles are
calibrated estimates derived from the motors, not independent encoder readings.

## What the hand does with a command

The Feetech servos do not jump to a goal position. Each one runs its own trapezoidal trajectory toward it, limited by
its acceleration register (ACC). At ACC 150 that is 22.8 rad/s² at the motor, about 22-30 rad/s² at a finger or
thumb joint (via the motor-per-joint ratio in `calibration.yaml`), and about 12 at the wrist. A 10° chirp needs
more than that above about 1.8 Hz. The ramp sits behind a dead time of about 40 ms from the command write to the
motor encoder. Together, the two explain the recorded chirps to 0.1-0.5° RMS (up to 2.5 Hz). A linear model (a PD
joint with any stiffness and damping, plus a delay) cannot reproduce them. The goal current (900 mA) caps the
torque. A free-moving joint never reaches it, but a joint pushing against something does.

`config.yaml` sets the ramp (`servo_acceleration_raw`). Both the recorder and the ROS hand controller write it after
`init_joints`, read it back, and refuse to run if a motor does not hold it (`src/hand_control/orca/servo_profile.py`).
Chirps and deployment therefore run the same ramp, and each run records it in `metadata.json`. If you change it,
record new chirps and refit before you deploy.

## Inspect and fit

Plot one recording (commanded/measured and Bode response). Prefix `MPLBACKEND=Agg` on a headless machine:

```bash
python experiments/orca/plot.py logs/chirp_v2/<run>/chirp_index_mcp_*.csv --save
```

Fit the servo model (dead time + ramp per joint, about 30 s, no Isaac):

```bash
python experiments/orca/fit_servo.py logs/chirp_v2/<run>_good                  # every joint in the run
python experiments/orca/fit_servo.py logs/chirp_v2/<run>_good --joints ring_abd # some joints
```

It writes `servo_fit.json` and `servo_fit.png` into the run folder and prints:

- per joint: the dead time, the fitted ramp, the ramp that ACC and the calibration predict (only with a
  `calibration.yaml` in the run folder, so from `chirp_all.py` runs), and the RMS error of the servo model
  and of the best linear model;
- flags: `clipped` (the joint stops short of the command, so the fit measures contact, not the free actuator) and
  `ramp differs` (more than 10% from the prediction). Flagged joints are left out of the sim values;
- the sim values: `ORCA_V2_SERVO_ACCEL` per joint group, and the actuator's dead time + lag for the fingers/thumb
  and the wrist.

The chirp is scored up to 2.5 Hz (`--max-chirp-hz`). Above that the ramp saturates completely and the servo runs
bang-bang: a 0.5% change in the ramp changes the trajectory by degrees, so no model can be scored there.

Stiffness, damping, friction and the effort limit cannot be identified from free-motion chirps. Once the ramp and the
dead time are in the model, stiffness 30, 60 and 120 with the same damping/stiffness ratio give the same response. In
the sim the joint behaves as a first-order lag with time constant D/K, so only D/K matters for these data. Set it so
that `delay_substeps / 120 s + D/K` equals the fitted dead time + lag.

For the faive_lab side, see the [faive_lab SysID guide](https://github.com/srl-ethz/faive_lab/blob/rwr2026/scripts/sysid/README.md)
and "The hand" in the [main README](../../README.md).

## Troubleshooting

- **Serial port not found or wrong device:** list the devices and set `port` in `config.yaml`:
  `python -m serial.tools.list_ports`.
- **"A complete calibration.yaml ... is required":** calibrate first. For `chirp.py`, check that `--config` points
  at a real folder that holds both files (see Setup).
- **"center +/- amplitude or step exceeds ROM":** lower `--amplitude-deg` or the largest `--steps-deg`, or move
  `--center-deg`. In minimum-gap mode, also lower `--abd-gap-deg`.
- **"Missed a full control period":** the bus is too slow for the commands. Close other programs on the port, or
  lower `--read-hz`.
- **Old runs with long CSV names:** `rename_chirp_results.py <run>` previews shorter names; `--apply` renames them.
