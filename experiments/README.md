## ORCA v2 hand (RWR 2026)

### Run a faive_lab policy on the hand

`rl_inference/policy_inference_cube_spin_v2.py` runs a policy trained in faive_lab (task `orca-v2-inhand-repose-v0`) on the real hand. Use the TorchScript file that faive_lab's `play.py` exports: `logs/rsl_rl/orca_v2_repose/<run>/exported/policy.pt`.

1. Start the hand controller (see [src/hand_control/README.md](../src/hand_control/README.md)):
    ```bash
    ros2 launch hand_control orcahand_v2_controller_launch.yaml hand_chirality:=right launch_rviz:=true
    ```
2. In a second terminal, start the policy. It commands the hand right away:
    ```bash
    ros2 run experiments policy_inference_cube_spin_v2.py --ros-args -p policy_ckpt_path:=/path/to/exported/policy.pt
    ```
    To check a new policy without moving the hand first, add `-p dry_run:=true`: it then only computes and logs the actions.
3. Stop with Ctrl+C; the hand holds its last commanded pose. For an emergency stop: `ros2 service call /right/hand/estop std_srvs/srv/SetBool "{data: true}"`.

Useful parameters (add each as `-p name:=value`):

| Parameter | Default | Meaning |
|---|---|---|
| `dry_run` | `false` | `true`: compute and log actions, but do not command the hand |
| `goal_ang_vel` | `1.0` | commanded spin speed about z in rad/s; the sign sets the direction |
| `goal_flip_period_s` | `0.0` | if > 0, flip the spin direction every this many seconds |
| `csv_log_path` | `""` | if set, log observations, joint commands and the cube pose to this CSV file |
| `policy_hz` | `20.0` | policy rate; must match the training task |

The joint limits, observation order and scales in the script must match the faive_lab task (`orca_v2_repose_env_cfg.py`).

### Evaluate policies on the real hand

`rl_inference/eval_policies.py` runs exported policies one after another on the real hand and counts the cube's turns with the camera. It takes policies without the cube pose (task `orca-v2-inhand-repose-v0`, 74 observations) and, with a camera calibration, policies with it (task `orca-v2-inhand-repose-extero-v0`, 82 observations); it tells them apart by their input size. One command starts the hand controller, the camera and the live window. It works on Linux and macOS.

For policies with the cube pose, the script needs the camera's pose relative to the hand: `--calibration` (the `cam_T_tag` file from `apriltag_calibration.py`, made with the camera where it is now) and `--hand-to-tag X Y Z ROLL PITCH YAW` (the `hand_to_tag_*` values of `cube_spin_v2.launch.py`). Check them with `--camera-only --calibration ...`: with the cube in the palm, the printed position is a few cm from (0, 0, -0.05), +z up. Such runs also write `cube_obs.csv`, and `results.csv` gets `cube_obs_tracked_fraction`, the share of policy steps with a cube pose at most `--cube-pose-timeout` (0.25 s) old.
```bash
python experiments/rl_inference/eval_policies.py --models-dir models_extero \
    --calibration ~/camera_extrinsics.npy --hand-to-tag X Y Z ROLL PITCH YAW
```

**Before the first use**
- The ROS workspace is built and sourced, and the hand is calibrated.
- The detector's model files are downloaded: `rl_inference/pose_estimation/download_models.sh`.
- An OAK camera looks down at the palm from above, on a USB 3 port. The script reads the camera itself, so the camera node must not run.

**Run**
```bash
# only the camera view: check that the cube is found (the hand is not started)
python experiments/rl_inference/eval_policies.py --camera-only

# only move the hand to the start pose and switch the torque off
python experiments/rl_inference/eval_policies.py --pose-only

# evaluate two policies, 60 s each at their goal speed (from W&B, else 2 rad/s)
python experiments/rl_inference/eval_policies.py models/ibex_policy.pt models/crow_policy.pt

# ... at one speed for both
python experiments/rl_inference/eval_policies.py --speed 2.0 models/ibex_policy.pt models/crow_policy.pt

# ... and upload each run to W&B (rwr-skin/real_hand_eval)
python experiments/rl_inference/eval_policies.py --wandb models/ibex_policy.pt models/crow_policy.pt

# download the new policies from W&B (orca_proprio -> models_proprio/, orca_extero -> models_extero/)
python experiments/rl_inference/download_policies.py
# list the models in models_proprio/ that have no run on W&B yet; Enter evaluates and uploads them
python experiments/rl_inference/eval_policies.py --dry-run
python experiments/rl_inference/eval_policies.py
```

**The hand moves when the script starts**: to its neutral pose, then to the start pose (the neutral pose with the wrist flexed: 2° before the upper end of its range, `--start-wrist-deg` sets another angle). For each policy, the script then:

1. switches the torque off and waits until the cube lies still in the palm and the motors are cool enough (see below) and at most 42.5 °C (a cold start, which the leaderboard requires), then counts down 3 s,
2. runs the policy for 60 s (`--duration`) of policy time at the model's goal speed: the speed of its runs on W&B (e.g. 1.5 rad/s for ibex), else 2 rad/s; `--speed` sets one for all (the sign is the direction). The clock stands still during a pause:
   - **The cube fell** (more than 6 cm from where it rested, or not seen for 3 s): the hand moves to the start pose and the torque goes off. Put the cube back; the run continues by itself when the cube lies still again. The space bar continues at once. After more than 2 drops (`--max-drops`) the run ends and the next policy starts.
   - **A motor reached 65 °C** (`--temp-limit`): the torque goes off where the hand is, without a movement. The run continues when the motors are cool enough for the rest of the time.
3. moves back to the start pose, switches the torque off and writes the results.

Keys in the camera window: space = start or continue now, `n` = next policy, `q` = quit. Ctrl+C in the terminal stops everything and switches the torque off.

**When is a motor cool enough?** A run (or the rest of it) starts when every motor is expected to stay 2 °C below the limit until the end: temperature + heating rate × remaining seconds. The heating rate of each motor is the largest of its last 8 runs (`thermal_history.json` in the results folder). Until a motor has 3 runs, 0.25 °C/s is assumed, so a 60 s run starts below 43 °C. The window shows the mean and the hottest motor; the limit applies to the hottest one, because a single hot motor is what fails.

**The window and the video** show the detected cube, the path of its centre over the last 8 s, where it rested (with the drop radius), the model, the policy time, the turns, the mean speed over the last 5 s, the drops and the motor temperatures. Turns are counted about the camera's viewing axis, positive in the direction of a positive `--speed`. The video leaves out cooling pauses.

**Results** go to `logs/real_eval/` (`--out`):
- `results.csv`: one row per run with the model, the policy file's hash, the git commit, the goal speed, the turns and mean speed of the whole run, the turns, time and mean speed before the first drop, the turns after 30 s, the fastest 10 s and the longest stall (these three before the first drop, defined as in the other real-hand evaluation), the drops and pauses, how well the cube was seen (`tracked_fraction`, `rejected_share`, `longest_gap_s`, `gap_s`), how far the cube moved from its resting place, the camera's tilt and whether it moved, the temperatures (mean and hottest, start and end, the fastest heating motor), the joints' delay and error against their commands, the motor currents, the lowest supply voltage and the servo faults.
- one folder per run: `video.mp4`, `trace.csv` (every detection: pixel and position of the cube's centre, orientation), `turns.csv` (the turns at every counted detection over policy time), `joints.csv` (commanded and measured angles and currents, 100 Hz), `joint_tracking.csv` (delay, error and current per joint), `temperatures.csv` and `health.csv` (per motor, once per second, with the policy time), `meta.json`.

**W&B** (`--wandb`, or `--auto`; set up once with `wandb login`, see the [installation](../README.md#installation-linux)): every finished run (`completed` or `too many drops`; not skipped, aborted or failed ones, and never a `--mock` run) goes to [rwr-skin/real_hand_eval](https://wandb.ai/rwr-skin/real_hand_eval) in the format of the other real-hand evaluation (`real_eval/upload_wandb.py` on `feat/real-hand-evaluation`), by `rl_inference/cube_eval/wandb_results.py` in a background process (log: `node_logs/wandb_upload.log`):
- one W&B run per run, job type `real_hand`, grouped and named by the model: the name W&B already has for it (e.g. `48_ibex`), else the file name without `_policy.pt`,
- config: `model.*`, `train.*` and `sim.*` (taken over from an earlier W&B run of the same model), `eval.*` (settings and conditions; `eval.evaluator = eval_policies.py`), `code.*`,
- summary: `result/*`, `cube/*`, `hand/*` (the other evaluation's keys where a figure means the same). The other evaluation ends a run at the first drop, so `result/turns`, `result/duration_s`, `result/rad_per_s`, `result/turns_30s`, `result/best_10s_rad_per_s` and `result/longest_stall_s` are up to the first drop; the whole run is in `result/turns_total`, `result/duration_total_s` and `result/rad_per_s_total`. Curves over the whole run's policy time (`curve/turns`, `curve/max_temp_c`, `curve/total_current_a`); the run folder's CSV files and `meta.json`; `video.mp4` as a playable `wandb.Video` (`eval/video`, in the run's media panel) unless `--no-wandb-video` (`--upload` also adds it to runs already on W&B without it),
- the `leaderboard` run, rebuilt from all runs on W&B of both evaluations, as the other evaluation builds it: per model, the runs at the model's own goal speed with a cold start (hottest motor at most 42.5 °C). A model's goal speed is the speed of its runs on W&B; a new model's first runs set it.

Without policy files (or with `--auto`) the script evaluates the `*_policy.pt` files in `models_proprio/` (`--models-dir`) that W&B has no `real_hand` run of: none of the same file (SHA-256) and none of the same model name. It leaves out files that do not take the 74 proprioceptive observations, and waits for Enter before it starts. `--dry-run` only prints the list. A model name as argument (`walrus`) means `models_proprio/walrus_policy.pt`. A run that is skipped (`n`) is not uploaded, so the model stays on the list. `--upload RUN_DIR ...` uploads saved run folders later (runs already on W&B are skipped); `python experiments/rl_inference/cube_eval/wandb_results.py untested models/` prints the list without the evaluation's dependencies.

A warning sign in a row: `tracked_fraction` well below 1, a `rejected_share` above about 0.1, or `gap_s` (the time in stretches of more than 0.6 s without a usable detection) above a few seconds. The turns are then less certain. On a Mac the detector reaches about 5 poses per second, which is enough up to about 3 rad/s; with an NVIDIA GPU it is faster.

`--mock` runs the whole flow without hardware (a drawn cube and a fake hand; `--mock-drop-after 5` makes the cube fall once). `python -m pytest tests/test_cube_eval.py` tests the turn counting, the drop detection, the temperature rule and what goes to W&B (without W&B).

The hand controller publishes the motors' health in joint order: `/right/hand/motor_current` (mA, with every joint state), and once per second `/right/hand/motor_temperature` (°C), `/right/hand/motor_voltage` (V) and `/right/hand/motor_status` (the servos' status bytes: 0x01 input voltage, 0x02 angle sensor, 0x04 overheating, 0x08 overcurrent, 0x20 overload).

### Detect the cube

The cube detector in `rl_inference/pose_estimation` finds the cube's corners in the OAK-D image and computes the cube pose from them (PnP).

0. Download the model files once: `rl_inference/pose_estimation/download_models.sh`.
1. Set your camera's ID (printed on the camera) in `cfgs/cams/oakd_cube_task.yaml`, and connect the camera to a USB 3 port with a USB 3 cable.
2. Start the camera, the cube detector and RViz:
    ```bash
    ros2 launch experiments cube_spin_v2.launch.py model_path:=/path/to/pose_estimation/finetuned_models/maskrcnn-finetuned_lightning_new_adamw.ckpt
    ```
3. Check the detection: in RViz, the cube should follow the real cube when you move and turn it. `ros2 run tf2_ros tf2_echo oakd_bottom_view_frame cube` prints the cube position relative to the camera; its distance should match a ruler measurement.

### Calibrate the camera

To get the cube pose relative to the hand, the camera's position is measured once against an AprilTag (`tag36h11`). Calibrate again whenever the camera or the tag moves.

1. Place the tag where the camera sees it, and measure the size of its black square (without the white border).
2. Start the camera (without `model_path:=`, the cube detector is not started):
    ```bash
    ros2 launch experiments cube_spin_v2.launch.py
    ```
3. In a second terminal, from the `experiments` folder, run the calibration with your tag's ID and size (in metres). Hold the camera and tag still for about 3 s while it collects 30 detections:
    ```bash
    python3 rl_inference/apriltag_calibration.py --tag-id 9 --tag-size 0.08 --output ~/camera_extrinsics.npy --visualize
    ```

### Set the hand's position

The launch places the hand relative to the tag with the `hand_to_tag_*` arguments: the tag's pose seen from the mount, i.e. the centre of the forearm's flat bottom face (frame `mount` in `orcahand_v2.urdf.xacro`). The mount's axes: x from the index finger towards the little finger, y out of the palm, z up the forearm.

The defaults (`0 -0.04 0.423`, `60 0 0`) are those of the extero evaluation setup: the tag (ID 9, 0.08 m) flat on the table beyond the fingertips, the forearm 30° below horizontal. How they were measured: [`cfgs/calibration/camera_extrinsics_2026-11-13.yaml`](cfgs/calibration/camera_extrinsics_2026-11-13.yaml). For another setup, measure them as follows.

1. Measure `hand_to_tag_x/y/z` (in metres) from the mount to the tag's centre, along the mount's axes. The mount is hidden under the forearm: it is 4.65 cm in from the index and little-finger sides, and 4.8 cm in from the palm and back sides.
2. Find `hand_to_tag_roll/pitch/yaw` (in degrees): how the tag's axes (x red, y green, z blue into the tag, as drawn by the calibration, or live by `python3 rl_inference/show_tag_axes.py` with the camera running) are turned relative to the mount's axes; roll about x, then pitch about the new y, then yaw about the new z. With the tag square to the hand, these are multiples of 90°. For example, a tag lying flat in front of the palm with its x along the mount's x: roll 180, pitch 0, yaw 0.
3. Launch with your values, e.g.:
    ```bash
    ros2 launch experiments cube_spin_v2.launch.py calibration:=$HOME/camera_extrinsics.npy \
        hand_to_tag_x:=0 hand_to_tag_y:=0.15 hand_to_tag_z:=0 hand_to_tag_roll:=180 hand_to_tag_pitch:=0 hand_to_tag_yaw:=0
    ```
4. Check: `ros2 run tf2_ros tf2_echo mount tag_frame` prints your x/y/z, and in RViz the hand model stands where the real hand is.


### Troubleshooting

**Camera**
- `oakd camera <ID> not available` in the log: wrong ID in `oakd_cube_task.yaml`, no udev rule, or the camera was still resetting from the previous run (wait about 3 s and launch again).
- `StereoDepthProperties has no attribute 'MedianFilter'`: depthai 3 is installed; run `pixi run pip install "depthai<3"`.
- `ros2 topic hz /oakd_bottom_view/color` shows images, but RViz does not: use an **Image** display (not Camera) with 'Reliability Policy: Best Effort'.

**Hand controller**
- The hand is not found or the serial port is not accessible: check the USB cable and add yourself to the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and in again).
- Joint positions look wrong after rebuilding: `calibration.yaml` lives in `install/` (see [src/hand_control/README.md](../src/hand_control/README.md)); calibrate again.

**Policy**
- `RuntimeError: The size of tensor a (74) must match the size of tensor b (81)` at startup: the policy expects a different observation than the script builds (e.g. a policy trained with the cube pose). `OBS_DIM` and the observation in the script must match the training task.

**Cube detection**
- `checkpoint.pth.tar not found` when the detector starts: the model files are not downloaded yet; run `experiments/rl_inference/pose_estimation/download_models.sh`.
