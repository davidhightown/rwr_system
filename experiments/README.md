## ORCA v2 hand (RWR 2026)

### Run a faive_lab policy on the hand

`rl_inference/policy_inference_cube_spin_v2.py` runs a policy trained in faive_lab on the real hand. Use the TorchScript file that faive_lab's `play.py` exports: `logs/rsl_rl/orca_v2_repose/<run>/exported/policy.pt`.

By default, the node expects a policy with the cube pose (task `orca-v2-inhand-repose-extero-v0`, 82 observations), see [Run the full system with the cube pose](#run-the-full-system-with-the-cube-pose). For a policy without it (task `orca-v2-inhand-repose-v0`, 74 observations), add `-p use_cube_pose:=false` in step 2 below.

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
| `use_cube_pose` | `true` | `false`: a policy without the cube pose (74 observations) |
| `cube_pose_timeout_s` | `0.25` | an older cube pose counts as lost; must match `timeout_s` of `CUBE_TRACKER` in faive_lab |

The joint limits, observation order and scales in the script must match the faive_lab task (`orca_v2_repose_env_cfg.py`).

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
    python3 rl_inference/apriltag_calibration.py --tag-id 9 --tag-size 0.075 --output ~/camera_extrinsics.npy --visualize
    ```

### Set the hand's position

The launch places the hand relative to the tag with the `hand_to_tag_*` arguments: the tag's pose seen from the mount, i.e. the centre of the forearm's flat bottom face (frame `mount` in `orcahand_v2.urdf.xacro`). The mount's axes: x from the index finger towards the little finger, y out of the palm, z up the forearm.

1. Measure `hand_to_tag_x/y/z` (in metres) from the mount to the tag's centre, along the mount's axes. The mount is hidden under the forearm: it is 4.65 cm in from the index and little-finger sides, and 4.8 cm in from the palm and back sides.
2. Find `hand_to_tag_roll/pitch/yaw` (in degrees): how the tag's axes (x red, y green, z blue into the tag, as drawn by the calibration) are turned relative to the mount's axes; roll about x, then pitch about the new y, then yaw about the new z. With the tag square to the hand, these are multiples of 90°. For example, a tag lying flat in front of the palm with its x along the mount's x: roll 180, pitch 0, yaw 0.
3. Launch with your values, e.g.:
    ```bash
    ros2 launch experiments cube_spin_v2.launch.py calibration:=$HOME/camera_extrinsics.npy \
        hand_to_tag_x:=0 hand_to_tag_y:=0.15 hand_to_tag_z:=0 hand_to_tag_roll:=180 hand_to_tag_pitch:=0 hand_to_tag_yaw:=0
    ```
4. Check: `ros2 run tf2_ros tf2_echo mount tag_frame` prints your x/y/z, and in RViz the hand model stands where the real hand is.


### Run the full system with the cube pose

Do the steps in [Detect the cube](#detect-the-cube), [Calibrate the camera](#calibrate-the-camera) and [Set the hand's position](#set-the-hands-position) first. Use the Linux robot PC with the NVIDIA GPU: the detector must reach the policy's 20 Hz.

```bash
ros2 launch experiments cube_spin_v2.launch.py \
    calibration:=$HOME/camera_extrinsics.npy \
    model_path:=/path/to/pose_estimation/finetuned_models/maskrcnn-finetuned_lightning_new_adamw.ckpt \
    policy:=/path/to/exported/policy.pt \
    enable_hand_controller:=true dry_run:=true csv_log_path:=$HOME/cube_dry_run.csv \
    hand_to_tag_x:=... hand_to_tag_y:=... hand_to_tag_z:=... \
    hand_to_tag_roll:=... hand_to_tag_pitch:=... hand_to_tag_yaw:=...
```

The launch file also takes `use_cube_pose`, `goal_ang_vel` and `goal_flip_period_s`.

- The policy starts at the first cube pose, and it commands nothing until then. So start without a cube in view, or with the cube already in the palm: **without `dry_run`, the hand moves at the first detection.**
- While the cube is lost, the policy gets the last pose with the flag `tracked` = 0, as in the simulated tracker.
- In the dry run, check the log line of the node: with the cube in the palm, `pos` is in the frame of the simulation, a few centimetres from the commanded position (0, 0, -0.05), and it follows the cube when you move it (+z is up). The CSV columns `cube_tracked` and `cube_age_s` show how often the cube is lost and how old the pose is. Compare them with `CUBE_TRACKER` in faive_lab.

### Evaluate policies on the real hand

`rl_inference/run_real_evaluation.sh` tests policies one after another on the real hand and counts the cube's turns with the camera. It is for policies **without the cube pose** (task `orca-v2-inhand-repose-v0`, 74 observations, exported as in [Run a faive_lab policy on the hand](#run-a-faive_lab-policy-on-the-hand)). One command starts everything: the camera with the cube detector, the hand controller, a window with the live view (topic `/real_eval/view` in `rqt_image_view`) and, for each run, the policy node. The camera calibration is not needed, because the turns are counted in the camera's frame.

**Before the first use**
- The ROS workspace is set up and built (see the [main README](../README.md)). The script expects the pixi workspace to be the folder that contains `src/rwr_system`, e.g. `~/ros_ws`; set `ROS_WS=/path/to/ros_ws` if yours is elsewhere. It starts everything through `pixi run` itself.
- The hand is calibrated (see [src/hand_control/README.md](../src/hand_control/README.md)).
- The detector's model files are downloaded: `rl_inference/pose_estimation/download_models.sh`.
- The camera is connected and looks down at the palm. The camera named in `cfgs/cams/oakd_cube_task.yaml` is used; if that one is not connected, the only connected OAK-D is used instead. `--camera-id <ID>` chooses one (the camera's log lists the connected IDs).

**1. List the policies.** Put each `policy.pt` in its own folder (any name), and list the folders in `models.json` in the parent folder. The default parent folder is `~/policies/real_test`; `--models-dir` sets another.
```json
[
  {"idx": 7, "name": "koala", "dir": "/home/me/policies/real_test/07_koala", "goal": 1.5},
  {"idx": 6, "name": "wombat", "dir": "/home/me/policies/real_test/06_wombat", "goal": -1.0}
]
```
- Without `--models`, the models are tested in the order of the file (`--oldest-first` reverses it).
- `goal` is the commanded spin speed in rad/s; the sign is the direction. `--speed` uses one speed for all models.
- `--models 06 koala` tests only these, in this order: the number is `idx` (`6` or `06`), or the `name`.

**2. Check the camera.** From the repository folder (`rwr_system`), run:
```bash
experiments/rl_inference/run_real_evaluation.sh --live
```
Only the camera starts; the hand does not move. Turn the cube by hand in front of the camera. The terminal shows the turns (counterclockwise seen from above counts up) and the poses per second, and the live view draws the detected cube. If the count runs backwards, the camera looks the other way: add `--axis-sign 1` to every command. Stop with Ctrl+C.

**3. Run the evaluation.** Keep the palm empty, then run:
```bash
experiments/rl_inference/run_real_evaluation.sh
```
**The hand moves after 5 s** to the start pose (the config's neutral pose with the wrist at `--start-wrist-deg`, 30 deg) and the torque is then switched off. `--pose-only` only does this and exits. For each model, the script then:

1. asks `Put the cube in the palm. Enter = start, n = next model, q = quit` (not after `Enter = save and start the next model`: then the next model starts by itself),
2. waits until every motor is below 42 °C (`--start-temp`) and the cube lies still,
3. switches the torque on and runs the policy for 60 s (`--duration`). Press Enter to stop a run early. A run also ends when the cube falls, when it is not detected for 5 s, when a motor reaches 58 °C (`--abort-temp`), or when the camera, the hand controller or the policy stops. A stopped camera or hand controller is restarted,
4. moves the hand back to the start pose, switches the torque off, prints the result and asks `Enter = save and start the next model, r = run again, n = next model without saving, q = quit`. After a run that was interrupted (stopped by you, too hot, camera, controller or policy stopped), it asks `Enter = run again, n = next model, q = quit` instead.

Ctrl+C at any time stops everything and leaves the hand in the start pose with the torque off; a run that Ctrl+C stops is still written (outcome `aborted`, not saved). Closing the terminal does the same. If the script crashes, every node it started gets Ctrl+C, and the hand controller (started with `torque_off_on_exit:=true`) switches the torque off as it exits. Nodes left over from an earlier session are listed at the start, with the offer to stop them.

A motor temperature that jumps by more than 15 °C from one reading to the next, or lies outside 0..100 °C, is a bad reading of the servo and ignored (a new value read twice in a row counts); `temperature_glitches` in `results.csv` counts them.

**Results** go to `<models-dir>/eval_<date_time>/`, one folder per start of the script (a session); a session without any run is deleted at the end, unless an error stopped it. To continue an evaluation later, pass that folder with `--out` (the full path): the models already saved there are skipped, except the ones named with `--models`, which run again. `--note "45 mm cube"` stores a note with every run of the session.
- `<models-dir>/all_results.csv` collects every saved run of all sessions (column `session`): the table to compare the models.
- `results.csv` has one row per run (also the ones not saved), and `summary.md` has the saved runs. They list the turns (positive = counterclockwise seen from above, the direction of a positive `goal`), the mean speed, the turns after 30 s, the fastest 10 s, the longest time without progress, how much of the run the cube was tracked, the joints' delay and the joints that barely follow their commands, the highest motor temperature, the summed mean motor current (`total_current_a`, effort and heating) and the joint with the highest current, the lowest supply voltage, the servo faults during the run (`servo_errors`, e.g. `pinky_mcp: overload`) and while parking afterwards (`servo_errors_parking`), and warnings. A warning means the count may be wrong: the cube was tracked for less than half of the run (`--min-tracked`), it was not detected for more than 3 s (`--max-gap`), or many poses were rejected. Run such a run again with `r`.
- Each run has its own folder: `video.mp4` (the live view), `turns_vs_time.png`, `joints.png` (commanded and measured angle of every joint), `trace.csv` (cube poses), `joints.csv` (commanded and measured joints and the motor currents, 100 Hz), `joint_tracking.csv` (delay, error and current per joint), `temperatures.csv`, `motor_status.csv` (supply voltage and servo status bits per motor, once per second), `policy_log.csv` and `policy_node.log` (the policy node), and `meta.json` (git commit, policy file hash, the model's W&B run if `models.json` has it, options, note, start temperatures and the result). The session folder also holds `code_changes_<time>.diff` (the uncommitted changes of `rwr_system`, so that the commit plus this file is the code that ran) and `node_logs/` (camera, hand controller and viewer; a restarted node appends to its log).

**Options**
- `--exposure-us 5000` fixes the camera's exposure time for less motion blur (0, the default: auto exposure; `--iso` sets the sensitivity). Compare the poses per second with `--live`.
- `--help` lists all options.
- `rl_inference/real_eval/test_offline.py` tests the turn counting and the analysis without hardware: `pixi run python src/rwr_system/experiments/rl_inference/real_eval/test_offline.py` from the workspace folder.

### Troubleshooting

**Camera**
- `oakd camera <ID> not available` in the log: wrong ID in `oakd_cube_task.yaml` (or `camera_id:=`; the log lists the connected IDs), no udev rule, or the camera was still resetting from the previous run (wait about 3 s and launch again).
- `StereoDepthProperties has no attribute 'MedianFilter'`: depthai 3 is installed; run `pixi run pip install "depthai<3"`.
- `ros2 topic hz /oakd_bottom_view/color` shows images, but RViz does not: use an **Image** display (not Camera) with 'Reliability Policy: Best Effort'.

**Hand controller**
- The controller publishes `/right/hand/motor_current` (mA, 100 Hz, read with the positions) and, for Feetech servos, `/right/hand/motor_voltage` and `/right/hand/motor_errors` (status bits OR-ed over 1 s: 0x01 input voltage, 0x02 angle sensor, 0x04 overheating, 0x08 overcurrent, 0x20 overload). Voltage and status bits come from orca_core's last state packet (a private attribute); after an orca_core update without them, the log says `No motor voltage and status bytes` once and the controller runs on.
- The hand is not found or the serial port is not accessible: check the USB cable and add yourself to the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and in again).
- Joint positions look wrong after rebuilding: `calibration.yaml` lives in `install/` (see [src/hand_control/README.md](../src/hand_control/README.md)); calibrate again.

**Policy**
- `The policy does not take N observations` at startup: the policy expects a different observation than the script builds. A policy with the cube pose takes 82, a policy without it takes 74 (`use_cube_pose:=false`). The observation in the script must match the training task.
- `Waiting for the first cube pose` does not stop: there is no recent `hand_root` -> `cube` transform. Check `ros2 run tf2_ros tf2_echo hand_root cube`, the `calibration:=` path (without it, the launch file does not publish the camera's position) and that the detector sees the cube.

**Cube detection**
- `checkpoint.pth.tar not found` when the detector starts: the model files are not downloaded yet; run `experiments/rl_inference/pose_estimation/download_models.sh`.
