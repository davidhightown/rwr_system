# ORCA v2 chirp and joint data acquisition

`chirp.py` collects ORCA v2 data through `orca_core`'s
public joint-position API. It runs directly with Python on macOS/Linux.

## Install for standalone v2 acquisition

From the repository root, create a separate Python 3.10+ environment (for
example, Python 3.12):

```bash
python3.12 -m venv .venv.chirp
source .venv.chirp/bin/activate
python -m pip install orca_core==0.5.2 scipy
```

No editable installation of `faive_system` is necessary for this standalone
script. A separate environment avoids the conflict between `orca_core`'s
NumPy requirement and `faive_system`'s NumPy 1.26.4 pin. For an existing
Faive/ROS environment, see the installation notes in
[`src/hand_control/README.md`](../../src/hand_control/README.md).

On macOS, you can use the `rwr_jazzy` environment instead (see the
[main README](../../README.md)). It already contains `orca_core` and `scipy`.

## Prepare the hand

1. Tension and calibrate the correct physical hand using the existing v2
   workflow. Chirp requires complete calibration and will **not** automatically
   calibrate it. See the hand-control README above.
2. Prepare your hand's own `config.yaml` and place its `calibration.yaml`
   in the same directory. Pass your config path with `--config` in the
   commands below, replacing `/path/to/your/config.yaml` with your file.
   After a `colcon build --symlink-install`, the installed `config.yaml` is a
   symlink to the source folder, and `chirp.py` then looks for `calibration.yaml`
   in the source folder. Copy both files to a folder of their own and use that:
   ```bash
   mkdir -p ~/ros_ws/chirp_config
   cp -L ~/ros_ws/install/hand_control/share/hand_control/orca/configs/orcahand_v2/right/{config,calibration}.yaml ~/ros_ws/chirp_config/
   ```
   Copy them again after each new calibration or change of `config.yaml`.
3. Connect power and USB. Keep `port: auto` in your config for automatic
   detection. If it fails or selects the wrong device, set `port` manually
   to the actual serial device (see Troubleshooting).
4. Stop the ROS controller, GUI, tensioning tool, and any other program using
   the same serial port. Leave the hand unloaded with room to move.

## Record all joints

## What the hand does with a command

The Feetech servos do not jump to a goal position. Each runs its own trapezoidal trajectory toward it,
limited by its acceleration register (ACC). At ACC 150 that is 22.8 rad/s^2 at the motor, about
22-30 rad/s^2 at a finger or thumb joint (the motor-per-joint ratio in `calibration.yaml`), and about 12 at
the wrist. A 10° chirp needs more than that above about 1.8 Hz. The ramp sits behind a dead time of about
40 ms, from the command write to the motor encoder. Together the two explain the recorded chirps to
0.1-0.5° RMS (up to 2.5 Hz). A linear model (a PD joint with any stiffness and damping, plus a delay) cannot
reproduce them. The goal current (900 mA) caps the torque. A free-moving joint never reaches it, but a joint
pushing against something does.

orca_core writes its own ACC (150) in `init_joints`. `config.yaml` now sets it explicitly
(`servo_acceleration_raw`). Both the recorder and the ROS hand controller write it after `init_joints`, read
it back, and refuse to run if a motor does not hold it (`src/hand_control/orca/servo_profile.py`). Chirps
and deployment therefore always run the same ramp, and each run records it in `metadata.json`. If you
change it, record new chirps and refit before you deploy.

## Record all joints

`chirp_all.py` records all 17 joints in one run (about 8 minutes), so you can
fit one finger and use the other three to find outliers and measure the spread:

```bash
python experiments/orca/chirp_all.py            # plan only
python experiments/orca/chirp_all.py --execute  # move the hand
```

It uses the installed `~/ros_ws/install/.../right/config.yaml` and the
`calibration.yaml` beside it; pass `--config` for another workspace. You do not
need the copy from step 2 above. The sweep options are the same as for `chirp.py`, and
`--joints` limits the run to some joints.

By default each joint ramps to its first chirp sample over about 1 s. `--no-ramp` skips this: the
joint jumps there at full motor speed and the previous joint snaps back to neutral. `metadata.json`
records the choice under `ramp_to_start`. The initial move to neutral at startup still ramps.

**Neighbouring fingers.** With all other fingers at neutral, an abduction sweep runs into the next finger. On
the 2026-11 runs, ring_abd stopped about 6° short on the middle-finger side, and middle_abd on both sides.
During a finger's abduction sweep, the other fingers therefore spread outward. Each moves just enough to stay
`--abd-gap-deg` (default 8°) from the next finger inward, at the extremes of the sweep. If a neighbour would leave
its ROM, the sweep center moves toward the free side instead (for example, pinky_abd sweeps about +13°). The plan
prints the centers and the poses (`centers_deg`, `neighbour_pose`), so check them with a dry run. On a new hand,
check them once with a small amplitude. `--abd-gap-deg 0` restores neutral neighbours. This assumes positive
abduction points toward the pinky, which the script checks against the neutral pose. Flexion sweeps keep the
other fingers at neutral.

After the run, the table flags `stops short -` / `stops short +`: the joint fell more than 1.5° short of
the command on that side below 1 Hz, where the servo dynamics cost well under a degree. A shortfall on one side
means something stops it there. On the 2026-11 runs, the middle/ring/pinky PIPs fall short by a similar amount on
both sides, with the neighbours at neutral. That points at the joint's own load (the motor stalls at its current
limit) rather than a collision. The motor current in `fast_*.csv` shows which.

After each chirp, `chirp_all.py` steps the joint by ±2°, ±6° and ±12° about the center (0.6 s each,
`--steps-deg`, `--step-hold`). Small steps show the dead time and large steps the ramp, without a
fit. Between the 20 Hz commands both scripts read the hand at 100 Hz (`--read-hz`), the read rate of the
deployment controller, so the bus carries the same load.

`--finger` (default `middle`) only saves the finger meant for the fit in
`metadata.json`, under `fit_finger` and `fit_joints`, so you can still fit another finger later.
Before it moves, the script warns you if a joint's motor travel in `calibration.yaml` is more than
25% below the median of the same joint on all fingers, and asks whether to continue.

After the run it prints a table, opens `overview.png`, and asks whether the run
was good and for optional notes (or pass `--notes`). The answer is added to the folder name:

| Suffix | Meaning |
|---|---|
| `_good` | You answered `y` |
| `_bad` | You answered `n`: the run itself was disturbed. A genuine outlier is still a good run |
| `_untagged` | You pressed Enter or Ctrl+C at the prompt; rename the folder when you decide |
| `_interrupted` | You stopped the run with Ctrl+C; the partial data is kept |
| `_failed` | The run stopped with an error; the partial data is kept |

The table flags `low motion` (moved less than 75% of the commanded peak-to-peak range), `short travel`,
`too few samples`, `timing` and `no data`. `overview.png` shows each joint type in one column.

## Collect data

To record only some joints, use `chirp.py`. The minimum for whole-hand SysID is eight representative
joints: the wrist, four thumb joints, and three joints of one finger. faive_lab fits these eight
and reuses that finger's parameters for the other three fingers (nine more joints). This reduces
recording and fitting time; it assumes similar dynamics, not identical measured behavior.

```bash
python experiments/orca/chirp.py \
  --config /path/to/your/config.yaml \
  --joints wrist thumb_cmc thumb_abd thumb_mcp thumb_dip index_abd index_mcp index_pip \
  --ctrl-freq 20 --duration 20 --f0 0.1 --f1 5 --amplitude-deg 10 --execute
```

Each joint runs a 20-second logarithmic sweep from 0.1 to 5 Hz at 20 Hz
control, with ±10° about its ROM midpoint. Other joints hold their neutral
pose. Use `--center-deg` to change the center; the trajectory must fit within
the joint limits. Ctrl+C stops the experiment.

## Output

Each run creates a directory under `logs/chirp_v2/` (`--output-dir` to change):

```text
logs/chirp_v2/<timestamp>/
├── metadata.json
├── chirp_<joint>_*.csv   # one CSV per selected joint: the read right after each 20 Hz command
├── fast_<joint>_*.csv    # every read at --read-hz, with the motor's velocity (rad/s) and current (mA)
└── steps_<joint>_*.csv   # the steps, same columns as fast_ (with --steps-deg)
```

`metadata.json` also records the servo ramp each motor held (`servo_profile`) and the servos' own
position-loop gains (`servo_gains`).

`chirp_all.py` adds the suffix above to the folder, and also saves `config.yaml`, `calibration.yaml` and
`overview.png` in it. The timestamp in the folder name is in UTC. `metadata.json` includes the motor
temperatures at the start and end of the run.

CSV columns (seconds and degrees; 400 rows per joint with the settings above):

```text
timestamp,commanded_position,measured_position
```

Keep the original filenames and `metadata.json` together. Only use recordings
marked `complete` in metadata for SysID. On motor-only hands, measured angles
are calibrated motor-derived estimates, not independent encoder measurements.

## Inspect a recording

Plot the commanded/measured angles and Bode response:

```bash
python -m pip install pandas matplotlib
python experiments/orca/plot.py /path/to/recording.csv --save
```

The PNG is saved in the current directory. On a headless machine, prefix
the command with `MPLBACKEND=Agg`. Plotting does not change the CSVs or SysID
settings; see `plot.py --help` for trim and alignment options.

## Fit the servo model

```bash
python experiments/orca/fit_servo.py logs/chirp_v2/<run>_good
```

It fits the dead time and the ramp of every joint, in the time domain, in about 30 s without Isaac. It also
fits a linear model for comparison. It writes `servo_fit.json` and `servo_fit.png` into the run folder and
prints:

- per joint: the dead time, the fitted ramp, the ramp that ACC and the calibration predict, and the RMS error
  of the servo model and of the best linear model;
- flags: `clipped` (the joint stops short of the command: a neighbouring finger, a coupled tendon or the
  current limit, so it measures contact, not the free actuator) and `ramp differs` (more than 10 % from
  the prediction). Flagged joints are left out of the sim values. On the 2026-11-05 runs (recorded before
  the neighbour clearance existed), middle/ring abd and middle/pinky pip clip; the index finger does not;
- the sim values: `ORCA_V2_SERVO_ACCEL` per joint group, and the actuator's dead time + lag for the
  fingers/thumb and the wrist.

The chirp is scored up to 2.5 Hz (`--max-chirp-hz`). Above that the ramp saturates completely, and the
servo runs bang-bang: a 0.5 % change of the ramp then changes the trajectory by degrees, so no model can
be scored there.

Stiffness, damping, friction and the effort limit cannot be identified from free-motion chirps: once the
ramp and the dead time are in the model, stiffness 30, 60 and 120 with the same damping/stiffness ratio
give the same response. In the sim, the joint behaves as a first-order lag with time constant D/K, so only
D/K matters for these data. Set it so that `delay_substeps / 120 s + D/K` equals the fitted dead time + lag.

## Use the recording in faive_lab SysID

[faive_lab SysID guide](https://github.com/srl-ethz/faive_lab/blob/rwr2026/scripts/sysid/README.md)

For the fit commands, see "The hand" in the [main README](../../README.md).

## Troubleshooting

**Serial connection:** list devices, then set `port` in your config if auto
detection fails:

```bash
python -m serial.tools.list_ports
```

**Check the plan without motion:**

```bash
python experiments/orca/chirp.py \
  --config /path/to/your/config.yaml --joints index_mcp --dry-run
```

**Check one joint with a small amplitude:**

```bash
python experiments/orca/chirp.py \
  --config /path/to/your/config.yaml \
  --joints index_mcp --amplitude-deg 3 --execute
```
