# RWR System

Code for the [Real-World Robotics](https://rwr.ethz.ch/) class (RWR 2026) with the ORCA v2 hand:

- **Reinforcement learning:** run a policy trained in simulation on the real hand (cube spin), with camera-based cube detection.
- **Imitation learning:** teleoperate the hand (Rokoko glove) and the Franka arm, record demonstrations, and run the trained policy.

The code of the RWR 2024 class is kept under the git tag [`rwr2024`](https://github.com/srl-ethz/rwr_system/tree/rwr2024).

## Repository layout

| Folder | ROS package | Content |
|---|---|---|
| `src/hand_control` | `hand_control` | ORCA v2 hand controller (through [orca_core](https://pypi.org/project/orca_core/)), Franka gripper, tensioning |
| `src/ingress` | `ingress` | Sensor input: OAK-D and ZED cameras, Rokoko glove, webcam hand tracking (MediaPipe) |
| `src/retargeter` | `retargeter` | Maps the human hand keypoints to the hand's joint angles, with tools to tune and calibrate it |
| `src/logger` | `logger` | Records ROS topics during demonstrations and converts them to HDF5 |
| `src/viz` | `viz` | Robot models (ORCA v2, Franka, cube), RViz configs, joint visualization |
| `src/common` | | Shared Python utilities |
| `experiments` | `experiments` | Launch files, configs and scripts for the tasks |
| `tests` | `faive_test` | Retargeter replay test, ORCA v2 chirp tests |
| `third_party/franka_msgs` | `franka_msgs` | Franka message definitions |

The Python package is called `faive_system` (e.g. `from faive_system.src.common.utils import ...`).

## Installation (Linux)

1. Create the `src` folder of your ROS 2 workspace and clone the repository into it:
   ```bash
      mkdir -p ~/ros_ws/src
      cd ~/ros_ws/src
      git clone https://github.com/srl-ethz/rwr_system.git
   ```
2. Set up a ROS 2 Jazzy environment, e.g. with pixi following [Getting Started with RoboStack](https://robostack.github.io/GettingStarted.html). With pixi, prefix the commands below with `pixi run` and run them from `~/ros_ws`.
3. Install the ROS dependencies:
   ```bash
      pixi add pip ros-jazzy-rosidl-typesupport-c ros-jazzy-joint-state-publisher-gui ros-jazzy-xacro
      pixi run which pip   # should print ~/ros_ws/.pixi/envs/default/bin/pip or similar
   ```

      If `which pip` prints another pip, comment out the conda/mamba setup in `~/.bashrc` or `~/.zshrc`.
4. Install the Python dependencies:
   ```bash
      pixi run pip install -e './src/rwr_system[all]'
      # the ORCA v2 driver. --no-deps: orca_core asks for a newer numpy than this package pins, but works with it
      pixi run pip install --no-deps orca_core==0.5.2
   ```
5. Build the workspace and source it:
   ```bash
      cd ~/ros_ws
      pixi run colcon build --symlink-install
      source install/setup.bash
   ```

      With pixi, add the source step to `~/ros_ws/pixi.toml` instead:

      If the build fails after changes, remove `build/`, `install/` and `log/` and build again.
6. For the RL task, download the cube detector's model files once:
   ```bash
      ./src/rwr_system/experiments/rl_inference/pose_estimation/download_models.sh
   ```
7. For the real-hand evaluation's results on [W&B](https://wandb.ai) (project `rwr-skin/real_hand_eval`), log in once with your API key from [wandb.ai/authorize](https://wandb.ai/authorize). `wandb` comes with the `[all]` install of step 4:
   ```bash
      pixi run wandb login
   ```

      You need to be a member of the `rwr-skin` team: `pixi run python -c "import wandb; print(wandb.Api().viewer.teams)"` should list it.

Run nodes with `ros2 run <package> <file>` and launch files with `ros2 launch <package> <launch file>`.

### Installation (Apple Silicon/macOS)

The commands to run the system are the same as on Linux; the code selects the platform's libraries and GPU by itself. Only the installation differs:

1. Clone the repository as in step 1 above, then create the environment (ROS 2 Jazzy, PyTorch and the other dependencies) with [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):
   ```bash
      cd ~/ros_ws/src/rwr_system
      micromamba create -f environment-macos.yml
      micromamba activate rwr_jazzy
      pip install --no-deps -e .
   ```
2. Build the workspace and source it. In zsh, use `setup.zsh`:
   ```bash
      cd ~/ros_ws
      colcon build --symlink-install --packages-select ingress hand_control viz experiments
      source install/setup.zsh
   ```

      If the repository is not in `~/ros_ws/src`, add `--base-paths /path/to/rwr_system` to the build command. In each new terminal, run `micromamba activate rwr_jazzy` and the `source` line again.
3. For the RL task, download the cube detector's model files (step 6 above).
4. For the real-hand evaluation's results on W&B, log in once (step 7 above). An environment created before `wandb` was added to `environment-macos.yml` needs it installed first:
   ```bash
      micromamba activate rwr_jazzy
      pip install wandb   # only for an older environment
      wandb login
   ```

Differences to Linux:

- **DDS:** the environment sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, because FastDDS delivers no camera images with macOS's default UDP limits. All machines in a ROS network must use the same setting.
- **Cube detector:** it runs on the Mac's GPU (MPS). A benchmark of the model on an M4 reached about 5.5 Hz, far below the 20 Hz of the policy, so use the Linux robot PC for the full cube spin system. To check the rate with a cube in view: `ros2 topic hz /cube_detection_node/cube_markers`.
- **Serial port:** the Linux low-latency setting of the serial port may not apply on macOS. Check with `ros2 topic hz /right/hand/joint_position_from_motor` that the joint state rate is high enough. The hand's controller board appears as two `usbmodem` ports; `port: auto` selects the motor port.
- **Message libraries:** macOS removes `DYLD_LIBRARY_PATH` from nodes started through `#!/usr/bin/env python3`, which makes this workspace's message types (e.g. `ingress.msg`) fail to load with `invalid allocator` or `type_support is null`. A node that uses them must call `restore_library_path_on_macos()` from `faive_system.src.common.ros_utils` at the top of the script. It does nothing on Linux.

For the chirp recorder on macOS, see [experiments/orca/README.md](experiments/orca/README.md).

## The hand

Do these steps in order after the assembly, and repeat them from step 1 when the tendons are slack. Only one program can use the hand's serial port at a time, so stop each step before you start the next. Source the workspace first (see Installation). Run the `python` commands from the repository root.

Make sure that the servo wiring is correct before you connect the power: the servos have no reverse-polarity protection.

1. **Tension the tendons.** The motors wind the tendons taut and then hold them under current. With the 3D-printed ratchet, turn the top spool of each servo clockwise until the tendon feels firm, and no further: too much tension makes joints stall before their end stops in calibration. Press Ctrl+C when done:
   ```bash
      python src/hand_control/orca/tensioning_v2.py
   ```

      `Motor ID N has not been fully calibrated` for all motors is normal here: this script reads the source config folder, and the calibration is in `install/`. Tensioning does not use it.
2. **Calibrate the hand** once after tensioning. The joints move to their end stops, so keep your fingers clear:
   ```bash
      ros2 launch hand_control orcahand_v2_controller_launch.yaml hand_chirality:=right calibrate:=true launch_rviz:=false
   ```

      The result is saved as `calibration.yaml` next to the installed config, so deleting `install/` means calibrating again. When it is finished, press Ctrl+C and compare the measured motor travel of equivalent joints:

      `calibrated: true` does not show that the result is good. A joint far below its neighbours stopped before its end stop (on our hand, the PIP joints measure about 90 to 100 degrees), see Troubleshooting in [experiments/README.md](experiments/README.md). Fix it and calibrate again before you continue.
3. **Move each joint** with the sliders and check that it moves correctly and over its full range:
   ```bash
      ros2 launch experiments orcahand_v2_joint_gui.launch.py
   ```

      Then stop the GUI.
4. **Record chirps for system identification.** The recorder uses `orca_core` directly, without ROS. Remove all loads from the hand and leave it room to move. See [experiments/orca/README.md](experiments/orca/README.md) for the details.
   ```bash
      # show the plan without moving
      python experiments/orca/chirp_all.py

      # test one joint with a small amplitude
      python experiments/orca/chirp_all.py --joints index_mcp --amplitude-deg 3 --execute

      # record all 17 joints, about 6 minutes
      python experiments/orca/chirp_all.py --execute

      # optional: no ramp to each joint's first chirp sample
      python experiments/orca/chirp_all.py --no-ramp --execute
   ```

      With `--no-ramp`, each joint jumps to its first chirp sample at full motor speed, and the previous joint snaps back to neutral, so test it on one joint with a small amplitude first. The recorded sweep is the same, so the fit does not change.

      **The default single-joint chirp** (the recorder's first version) is still `experiments/orca/chirp.py`: one joint (`index_mcp` by default, `--joints` for others), a logarithmic chirp from 0.1 to 5 Hz over 20 s, ±10° about the middle of the joint's range, commands at 20 Hz, the other joints at neutral, a ramp of about 1 s to the first sample and 1 s to settle:

      It writes `logs/chirp_v2/<UTC time>/chirp_index_mcp_*.csv` (`timestamp`, `commanded_position`, `measured_position`: one row per command, the file the fit reads) and `metadata.json`. Since then it also does the following, none of which changes the sweep:
   - **Servo ramp:** after start-up it writes `servo_acceleration_raw` from `config.yaml` (150) to every motor and records it. This is the same value `orca_core` 0.5.2 writes by itself, so the motors behave as in the first version.
   - **Fast reads:** between commands it reads the joint at 100 Hz, as the deployment controller does, into `fast_*.csv` with the motor's velocity and current. `--read-hz 20` turns this off, so the bus carries exactly the first version's traffic.
   - **Abduction joints** (`*_abd`): the neighbouring fingers spread so they do not touch the swept finger, and the sweep centre may move off the middle of the range. `--abd-gap-deg 0` keeps them at neutral and the centre in the middle, as in the first version.
   - **Optional extras, off by default:** step tests after the chirp (`--steps-deg`) and `--no-ramp`.

      After the run, the script shows a table and `overview.png`, and asks whether the run was good. Answer `n` only if the run itself was disturbed (bumped, loaded, loose cable): a genuine outlier is still a good run, and it is what you want to measure. The answer goes into the folder name, so `ls -d logs/chirp_v2/*_good` lists the good runs. Press Enter to decide later and rename the folder yourself, and do not tag test runs as good. Each folder has the `config.yaml` and `calibration.yaml` of that run, and its name has the start time in UTC. `logs/chirp_v2/` is ignored by git, so copy the runs somewhere safe. See [experiments/orca/README.md](experiments/orca/README.md).
5. **Fit the joint parameters** on the training computer, in the Python environment with Isaac Lab, from `scripts/motor_param_characterization` in `rwr_faive_lab`. Install the fit's packages once with `python -m pip install scikit-optimize pandas`. The simulator's joint names have the prefix `right_`:
   ```bash
      for j in wrist thumb_cmc thumb_abd thumb_mcp thumb_dip middle_abd middle_mcp middle_pip; do
        python optimization.py --joint_name right_$j --ground_truth <run_dir>/chirp_${j}_*.csv
      done
   ```

      The output shows the best parameters, and `logs/optimizations/<joint>_<timestamp>/frf_comparison.png` compares them with the real hand. If a best value is at the limit of its range, make the range larger in `optimization.py` and fit again. If you change the chirp on the real hand, also change it in `chirp_characterization.py`.
6. **Use the parameters.** Write them into `source/faive_lab/faive_lab/assets/orca_v2.py` in `rwr_faive_lab` (the values of the finger you fit also apply to the other three fingers), then train the policy again.

If something fails, see Troubleshooting in [experiments/README.md](experiments/README.md).

To check the hand without teleop or a policy, `experiments/fixed_command.py` publishes fixed joint poses (edit the poses in the script), and `ros2 run experiments command_gui.py -n 17 -t policy_output` moves each joint with a slider.

## Reinforcement learning: cube spin

Run a policy trained in faive_lab on the hand, detect the cube, and calibrate the camera: see [experiments/README.md](experiments/README.md).

### Deploy and evaluate policies on the real hand

`experiments/rl_inference/eval_policies.py` runs exported policies one after another on the hand for 60 s each, counts the cube's turns with the camera from above, and uploads the results to W&B. One command starts the hand controller, the camera and a live window; do not start them yourself. Details (start pose, drops, heat pauses, all result columns): [experiments/README.md](experiments/README.md#evaluate-policies-on-the-real-hand).

1. **Download the new policies.** Training logs every exported policy as a W&B artifact in `rwr-skin/orca_proprio` or `rwr-skin/orca_extero`. This copies those not yet here to `models_proprio/<name>_policy.pt` and `models_extero/<name>_policy.pt` (with `<name>_policy.json` next to each: artifact, training run, checkpoint, commit). Files already there are left alone. Both folders are git-ignored.
   ```bash
      python experiments/rl_inference/download_policies.py             # --dry-run: only list the new ones
   ```

      The steps below evaluate the proprioceptive policies in `models_proprio/`. For the policies in `models_extero/`, see [Policies with the cube pose](#policies-with-the-cube-pose) below.
2. **Check the setup** (the hand is calibrated, the camera looks down at the palm on a USB 3 port, the W&B login of the installation is done):
   ```bash
      source ~/ros_ws/install/setup.zsh # source to get the handcontroller

      python experiments/rl_inference/eval_policies.py --camera-only   # the cube is found; the hand is not started
      python experiments/rl_inference/eval_policies.py --pose-only     # the hand goes to the start pose, torque off
   ```
3. **Evaluate every untested model.** Without arguments the script lists the models in `models_proprio/` with no run on W&B yet (none of the same file and none of the same name), sets aside those it cannot run (teachers and history students, which take other observations than the 74 proprioceptive ones), and after Enter evaluates the rest and uploads each run:
   ```bash
      python experiments/rl_inference/eval_policies.py --dry-run   # only the list: tested / untested
      python experiments/rl_inference/eval_policies.py
   ```

      To deploy specific models, name them (a name is looked up in `models_proprio/`; a path works too), and add `--wandb` to upload the runs: `python experiments/rl_inference/eval_policies.py --wandb walrus models/ibex_policy.pt`. Without `--wandb` nothing is uploaded for named models.
4. **Run.** Put the cube in the palm; each run starts by itself when the cube lies still and the motors are cool enough. If the cube falls, put it back: the run goes on. At the third drop the run ends and the next policy starts (`--max-drops 2`). Keys in the window: space = start or continue now, `n` = next policy, `q` = quit. Ctrl+C switches the torque off and stops.
5. **Results.** Locally in `logs/real_eval/` (`results.csv` and a folder per run with the video). On W&B in [rwr-skin/real_hand_eval](https://wandb.ai/rwr-skin/real_hand_eval), in the same format as the other real-hand evaluation (one run per policy run, grouped by model, and the `leaderboard` run), with `eval.evaluator = eval_policies.py`. The upload runs in the background while the next policy starts. Only finished runs are uploaded (not skipped or aborted ones). The video of each run is uploaded too (about 25 MB; `--no-wandb-video` leaves it out). To add the videos to runs already on W&B, upload their folders again: `python experiments/rl_inference/eval_policies.py --upload logs/real_eval/2026*_*`. If an upload failed (e.g. offline), the script prints the command to upload again, e.g. `python experiments/rl_inference/eval_policies.py --upload logs/real_eval/20261111_203512_lark`.

**Comparable with the other real-hand evaluation** (`evaluate_real_policies.py` on `feat/real-hand-evaluation`), whose runs are in the same W&B project and leaderboard:

- **Drops:** the other evaluation ends a run at the first drop; this one puts the cube back and continues. On W&B, `result/turns`, `result/duration_s`, `result/rad_per_s`, `result/turns_30s`, `result/best_10s_rad_per_s` and `result/longest_stall_s` are therefore counted up to the first drop, as there; the whole run's figures are `result/turns_total`, `result/duration_total_s` and `result/rad_per_s_total`, and `result/drops` counts the drops. `results.csv` has both.
- **Goal speed:** turns in 60 s depend on the commanded speed, so the leaderboard compares each model at its own goal speed only. Without `--speed`, each model runs at its goal speed on W&B (the speed of its runs there, e.g. 1.5 rad/s for ibex, 1.8 for dolphin); a model new to W&B runs at 2 rad/s, and its first runs set its goal speed. The script prints the speed of every model before it starts. `--speed` sets one speed for all; runs at another speed than the model's goal are on W&B but not in the leaderboard.
- **Cold start:** the leaderboard also takes only runs that started with the hottest motor at most 42.5 °C, so a run starts only below that.
- Left as they are: the start pose (wrist 33° here, 30° there), the camera exposure (5 ms here, automatic there), the heat rule (pause at 65 °C here, stop at 58 °C there) and the turn counting.

## Imitation learning

The hand follows the Rokoko glove through the retargeter, and the Franka arm follows the glove's wrist. The Franka arm controller is not part of this repository: it subscribes to `/right/arm/end_effector_pose_cmd` and publishes `/right/arm/end_effector_pose`.

1. **Cameras.** Set the ID printed on each OAK-D camera in a file in `experiments/cfgs/cams/` and select it with `OAKD_CAMERA_CFG_NAME` in `record_demonstration.launch.py` and `controllers_and_cameras.launch.py` (default: `oakd_scissors_task.yaml`). For a ZED camera, set `ZED_CAMERA_CFG_NAME`. The camera extrinsics can be measured with `experiments/calibrate_camera_pose.py`.
2. **Retargeter.** Tune the mapping from your hand to the robot hand, see [src/retargeter/README.md](src/retargeter/README.md). The ORCA v2 settings are in `experiments/cfgs/retargeter/`.
3. **Test the teleop** (no recording):
   ```bash
      ros2 launch experiments teleop_rokoko.launch.py
   ```

      A `keyboard_runner` terminal opens: press Enter to start, pause and resume the teleop. Foot pedals that type keys work the same way.
4. **Record demonstrations.** Set `DATASET_NAME` in `record_demonstration.launch.py`, then:
   ```bash
      ros2 launch experiments record_demonstration.launch.py
   ```

      In the `keyboard_runner` terminal, Enter starts the teleop, starts the recording and stops it. Each demonstration is saved as an `.h5` file in `tmp/<DATASET_NAME>/` (relative to where you launched). The recorded topics are listed in `experiments/cfgs/logger/datacollection_default.yaml`.
5. **Process the data.** Resample all topics of the recordings in a folder to a common rate:
   ```bash
      ros2 run logger syncronize_data.py tmp/<DATASET_NAME> --sampling_freq 10
   ```

      Check the demonstrations with `experiments/scripts/plot_trajectories.py` and `experiments/scripts/visualize_videos.py`.
6. **Train** a policy, e.g. with [srl_il](https://github.com/srl-ethz/srl_il).
7. **Run the policy.** Start the cameras and the hand, then the policy:
   ```bash
      ros2 launch experiments controllers_and_cameras.launch.py
      ros2 launch experiments run_policy_inference.launch.py policy_ckpt_path:=/path/to/policy.ckpt
   ```

      The inference config (`experiments/cfgs/inference_config_*.yaml`, set with `inference_cfg_file:=`) maps the camera and joint topics to the policy's inputs and outputs; it must match your training data.

## Other tools

| Tool | Use |
|---|---|
| `experiments/gesture_runner.py` | Enables the retargeter only while one wrist is held more than 10 cm higher than the other (needs both gloves), so the hand stops following when you lower it |
| `experiments/run_keyboard_teleop_franka.py` | Moves the Franka end effector with the keyboard |
| `experiments/calibrate_franka_load.py` | Calibrates the load mounted on the Franka |
| `src/viz/visualize_static_frames.py` | Publishes fixed frames (e.g. cameras) from a YAML file such as `experiments/cfgs/tf/static_frames_bimanual.yaml` |
| `experiments/orca/` | ORCA v2 chirp data acquisition for system identification, see [experiments/orca/README.md](experiments/orca/README.md) |

## Contributing

To contribute, fork the repository and open a pull request.
