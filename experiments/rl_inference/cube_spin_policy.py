"""The cube-spin policy without ROS: observation, policy call and action processing of the faive_lab in-hand tasks,
shared by policy_inference_cube_spin_v2.py and eval_policies.py.

The policy is the TorchScript file that faive_lab's play.py exports (it includes the observation normalisation). Its
number of inputs selects the observation layout (LAYOUTS):
    proprio    74  orca-v2-inhand-repose-v0, orca-v2-inhand-all-axes-vel-v0
    cube_pose  82  orca-v2-inhand-repose-extero-v0, orca-v2-inhand-all-axes-vel-extero-v0 (lemur)
    goal       84  orca-v2-inhand-goal-v0 (manta, DeXtreme goal orientations)

Each step:
1. Build the observation as in the simulation (USD joint order):
       joint_pos       joint positions normalised to [-1, 1] over the joint limits
       joint_vel       joint velocities * 0.2
       cube_pose       position (3), orientation w, x, y, z (4), tracked flag (1); cube_pose and goal only
       goal_command    [0, 0, -0.05, angular velocity about x, y, z]; proprio and cube_pose only
       goal_quat       goal orientation w, x, y, z (4); goal only
       goal_diff       rotation from the goal to the tracked orientation (4); goal only
       last_action     the policy's previous raw output
       last_joint_cmd  the previous joint position command (radians)
2. Run the policy, map its output onto the joint limits and smooth it with the same moving average as the
   simulation (target = alpha * new + (1 - alpha) * previous).

With the goal layout, the core samples the goals itself (OrientationGoalCommand in faive_lab): a new goal as soon as the
tracked orientation is within goal_tolerance of the current one.

The joint limits, observation order and scales below must match the training task (orca_v2_repose_env_cfg.py).
"""

import numpy as np
import torch

# Joint order of the v2 USD model in Isaac Lab (= order of the policy's observations and actions), with the joint
# limits the task normalises over (soft limits of the v2 USD, degrees).
JOINT_LIMITS_USD_DEG = {
    "wrist": (-65.0, 35.0),
    "index_abd": (-25.0, 30.0),
    "middle_abd": (-27.0, 27.0),
    "pinky_abd": (-30.0, 30.0),
    "ring_abd": (-27.0, 27.0),
    "thumb_cmc": (-45.0, 33.0),
    "index_mcp": (-25.0, 100.0),
    "middle_mcp": (-25.0, 100.0),
    "pinky_mcp": (-25.0, 100.0),
    "ring_mcp": (-25.0, 100.0),
    "thumb_abd": (-18.0, 55.0),
    "index_pip": (-15.0, 107.0),
    "middle_pip": (-15.0, 107.0),
    "pinky_pip": (-15.0, 107.0),
    "ring_pip": (-15.0, 107.0),
    "thumb_mcp": (-25.0, 100.0),
    "thumb_dip": (-15.0, 107.0),
}
NUM_JOINTS = len(JOINT_LIMITS_USD_DEG)
OBS_DIM = 17 + 17 + 6 + 17 + 17  # proprioception only
OBS_DIM_CUBE_POSE = 17 + 17 + 8 + 6 + 17 + 17
OBS_DIM_GOAL = 17 + 17 + 8 + 4 + 4 + 17 + 17
LAYOUTS = {OBS_DIM: "proprio", OBS_DIM_CUBE_POSE: "cube_pose", OBS_DIM_GOAL: "goal"}
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
GOAL_MODES = ("uniform", "face90")


def quat_mul(a, b):
    """Hamilton product of quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_unique(q):
    """Isaac Lab's quat_unique: the real part is not negative."""
    return (-q if q[0] < 0 else q).astype(np.float32)


def quat_angle(q):
    """Rotation angle of a unit quaternion (rad, 0 to pi)."""
    return float(2.0 * np.arccos(np.clip(abs(q[0]), 0.0, 1.0)))


class CubeSpinPolicy:
    """Joint positions and velocities in, joint position targets out, all in the hand driver's joint order (radians).

    goal_axis ("x", "y" or "z", axes of the simulation's env frame) and goal_ang_vel set the velocity command of the
    proprio and cube_pose layouts. goal_mode ("uniform" as in training, or "face90": each goal one 90 deg face turn from
    the last) and goal_tolerance (rad) set the goals of the goal layout.
    """

    def __init__(self, policy_path, joint_order, goal_ang_vel=1.0, action_alpha=0.2, velocity_obs_scale=0.2,
                 goal_axis="z", goal_mode="uniform", goal_tolerance=0.4, seed=None):
        self.joint_order = list(joint_order)  # the hand driver's order (config.yaml joint_ids)
        self.joint_order_usd = list(JOINT_LIMITS_USD_DEG)
        if set(self.joint_order) != set(self.joint_order_usd):
            raise ValueError(f"hand config joints {self.joint_order} do not match the policy's joints "
                             f"{self.joint_order_usd}")
        if goal_axis not in AXES:
            raise ValueError(f"goal_axis must be one of {list(AXES)}, not {goal_axis!r}")
        if goal_mode not in GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {GOAL_MODES}, not {goal_mode!r}")
        self.robot_to_usd_idx = [self.joint_order.index(name) for name in self.joint_order_usd]
        self.usd_to_robot_idx = [self.joint_order_usd.index(name) for name in self.joint_order]
        self.limits_usd = np.deg2rad(np.array(list(JOINT_LIMITS_USD_DEG.values()), dtype=np.float32))
        self.action_alpha = float(action_alpha)
        self.velocity_obs_scale = float(velocity_obs_scale)
        self.goal_command = np.array([0.0, 0.0, -0.05, *(float(goal_ang_vel) * np.array(AXES[goal_axis]))],
                                     dtype=np.float32)
        self.goal_mode = goal_mode
        self.goal_tolerance = float(goal_tolerance)
        self.rng = np.random.default_rng(seed)

        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        self.obs_dim, out = None, None
        for dim in LAYOUTS:
            try:
                with torch.no_grad():
                    out = self.policy(torch.zeros(1, dim))
            except RuntimeError:
                continue
            self.obs_dim = dim
            break
        if self.obs_dim is None:
            raise ValueError(f"{policy_path} takes none of the known observation layouts "
                             + ", ".join(f"{name} ({dim})" for dim, name in LAYOUTS.items()))
        if out.shape[-1] != NUM_JOINTS:
            raise ValueError(f"{policy_path} outputs {out.shape[-1]} actions, expected {NUM_JOINTS}")
        self.layout = LAYOUTS[self.obs_dim]
        self.uses_cube_pose = self.layout != "proprio"
        self.reset()

    def reset(self):
        """As at a reset in the task: last action zero; the moving average starts from the pose of the next step, the
        first goal is sampled at the next step."""
        self.last_action_usd = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.last_joint_cmd_usd = None
        self.goal_quat = None
        self.goals_reached = 0

    def flip_direction(self):
        self.goal_command[3:6] *= -1.0

    def next_goal(self, base_quat):
        """A new goal orientation (w, x, y, z), as OrientationGoalCommand._resample_command."""
        if self.goal_mode == "uniform":
            q = self.rng.normal(size=4)
            q = (q / np.linalg.norm(q)).astype(np.float32)
        else:
            axis = np.array(AXES[self.rng.choice(list(AXES))], dtype=np.float32) * self.rng.choice([-1.0, 1.0])
            turn = np.array([np.cos(np.pi / 4), *(np.sin(np.pi / 4) * axis)], dtype=np.float32)
            q = quat_mul(turn, base_quat)  # a turn about the env-frame axis
        return quat_unique(q)

    def goal_diff(self, object_quat):
        """Rotation from the goal to object_quat, as cube_pose_tracked_goal."""
        return quat_unique(quat_mul(object_quat, quat_conjugate(self.goal_quat)))

    def normalize(self, joint_pos_usd):
        """Isaac Lab's scale_transform (joint_pos_limit_normalized): limits -> [-1, 1], not clipped."""
        low, high = self.limits_usd[:, 0], self.limits_usd[:, 1]
        return (2.0 * (joint_pos_usd - (low + high) / 2) / (high - low)).astype(np.float32)

    def denormalize(self, action_usd):
        """Isaac Lab's unscale_transform (rescale_to_limits): [-1, 1] -> limits."""
        low, high = self.limits_usd[:, 0], self.limits_usd[:, 1]
        return (low + (action_usd + 1.0) / 2 * (high - low)).astype(np.float32)

    def step(self, joint_pos, joint_vel, cube_pose=None):
        """One policy step. cube_pose is (position (3), orientation w, x, y, z (4), tracked) in the simulation's frame,
        required by the cube_pose and goal layouts. Returns (target in the hand driver's order, observation, raw action,
        target in USD order), or None if the joint state is not finite (nothing changes then)."""
        joint_pos_usd = np.asarray(joint_pos, dtype=np.float32)[self.robot_to_usd_idx]
        joint_vel_usd = np.asarray(joint_vel, dtype=np.float32)[self.robot_to_usd_idx]
        if not (np.all(np.isfinite(joint_pos_usd)) and np.all(np.isfinite(joint_vel_usd))):
            return None
        if self.uses_cube_pose and cube_pose is None:
            raise ValueError(f"the {self.layout} layout ({self.obs_dim} observations) needs the cube pose")
        if self.last_joint_cmd_usd is None:
            self.last_joint_cmd_usd = joint_pos_usd.copy()

        if self.layout == "proprio":
            command_obs = [self.goal_command]
        else:
            object_pos, object_quat, tracked = cube_pose
            object_quat = np.asarray(object_quat, dtype=np.float32)
            cube_obs = [np.asarray(object_pos, dtype=np.float32), object_quat, np.array([float(tracked)], dtype=np.float32)]
            if self.layout == "cube_pose":
                command_obs = cube_obs + [self.goal_command]
            else:
                # the simulation resamples after the step, before the observation; a held (lost) pose reaches no goal
                if self.goal_quat is None:
                    self.goal_quat = self.next_goal(object_quat)
                elif tracked and quat_angle(self.goal_diff(object_quat)) < self.goal_tolerance:
                    self.goals_reached += 1
                    self.goal_quat = self.next_goal(self.goal_quat)
                command_obs = cube_obs + [self.goal_quat, self.goal_diff(object_quat)]

        obs = np.concatenate([
            self.normalize(joint_pos_usd),
            joint_vel_usd * self.velocity_obs_scale,
            *command_obs,
            self.last_action_usd,
            self.last_joint_cmd_usd,
        ]).astype(np.float32)

        with torch.no_grad():
            action_usd = self.policy(torch.from_numpy(obs).unsqueeze(0))[0].numpy().astype(np.float32)
        self.last_action_usd = action_usd

        # as the task's EMAJointPositionToLimitsAction
        target_usd = self.denormalize(np.clip(action_usd, -1.0, 1.0))
        target_usd = self.action_alpha * target_usd + (1.0 - self.action_alpha) * self.last_joint_cmd_usd
        target_usd = np.clip(target_usd, self.limits_usd[:, 0], self.limits_usd[:, 1]).astype(np.float32)
        self.last_joint_cmd_usd = target_usd
        return target_usd[self.usd_to_robot_idx], obs, action_usd, target_usd
