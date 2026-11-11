"""The cube-spin policy without ROS: observation, policy call and action processing of the faive_lab task
orca-v2-inhand-repose-v0 (proprioception only), shared by policy_inference_cube_spin_v2.py and eval_policies.py.

The policy is the TorchScript file that faive_lab's play.py exports (it includes the observation normalisation).
Each step:
1. Build the observation as in the simulation (USD joint order, 74 values):
       joint_pos       joint positions normalised to [-1, 1] over the joint limits
       joint_vel       joint velocities * 0.2
       goal_command    [0, 0, -0.05, 0, 0, angular velocity about z]
       last_action     the policy's previous raw output
       last_joint_cmd  the previous joint position command (radians)
2. Run the policy, map its output onto the joint limits and smooth it with the same moving average as the
   simulation (target = alpha * new + (1 - alpha) * previous).

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
OBS_DIM = 17 + 17 + 6 + 17 + 17


class CubeSpinPolicy:
    """Joint positions and velocities in, joint position targets out, all in the hand driver's joint order (radians)."""

    def __init__(self, policy_path, joint_order, goal_ang_vel=1.0, action_alpha=0.2, velocity_obs_scale=0.2):
        self.joint_order = list(joint_order)  # the hand driver's order (config.yaml joint_ids)
        self.joint_order_usd = list(JOINT_LIMITS_USD_DEG)
        if set(self.joint_order) != set(self.joint_order_usd):
            raise ValueError(f"hand config joints {self.joint_order} do not match the policy's joints "
                             f"{self.joint_order_usd}")
        self.robot_to_usd_idx = [self.joint_order.index(name) for name in self.joint_order_usd]
        self.usd_to_robot_idx = [self.joint_order_usd.index(name) for name in self.joint_order]
        self.limits_usd = np.deg2rad(np.array(list(JOINT_LIMITS_USD_DEG.values()), dtype=np.float32))
        self.action_alpha = float(action_alpha)
        self.velocity_obs_scale = float(velocity_obs_scale)
        self.goal_command = np.array([0.0, 0.0, -0.05, 0.0, 0.0, float(goal_ang_vel)], dtype=np.float32)

        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        with torch.no_grad():
            try:
                out = self.policy(torch.zeros(1, OBS_DIM))
            except RuntimeError as e:
                raise ValueError(f"{policy_path} does not take {OBS_DIM} observations (a policy with the cube pose "
                                 f"or of another task?): {e}") from e
        if out.shape[-1] != NUM_JOINTS:
            raise ValueError(f"{policy_path} outputs {out.shape[-1]} actions, expected {NUM_JOINTS}")
        self.reset()

    def reset(self):
        """As at a reset in the task: last action zero; the moving average starts from the pose of the next step."""
        self.last_action_usd = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.last_joint_cmd_usd = None

    def flip_direction(self):
        self.goal_command[5] *= -1.0

    def normalize(self, joint_pos_usd):
        """Isaac Lab's scale_transform (joint_pos_limit_normalized): limits -> [-1, 1], not clipped."""
        low, high = self.limits_usd[:, 0], self.limits_usd[:, 1]
        return (2.0 * (joint_pos_usd - (low + high) / 2) / (high - low)).astype(np.float32)

    def denormalize(self, action_usd):
        """Isaac Lab's unscale_transform (rescale_to_limits): [-1, 1] -> limits."""
        low, high = self.limits_usd[:, 0], self.limits_usd[:, 1]
        return (low + (action_usd + 1.0) / 2 * (high - low)).astype(np.float32)

    def step(self, joint_pos, joint_vel):
        """One policy step. Returns (target in the hand driver's order, observation, raw action, target in USD
        order), or None if the joint state is not finite (nothing changes then)."""
        joint_pos_usd = np.asarray(joint_pos, dtype=np.float32)[self.robot_to_usd_idx]
        joint_vel_usd = np.asarray(joint_vel, dtype=np.float32)[self.robot_to_usd_idx]
        if not (np.all(np.isfinite(joint_pos_usd)) and np.all(np.isfinite(joint_vel_usd))):
            return None
        if self.last_joint_cmd_usd is None:
            self.last_joint_cmd_usd = joint_pos_usd.copy()

        obs = np.concatenate([
            self.normalize(joint_pos_usd),
            joint_vel_usd * self.velocity_obs_scale,
            self.goal_command,
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
