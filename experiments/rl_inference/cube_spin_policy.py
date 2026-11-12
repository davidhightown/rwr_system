"""The cube-spin policy without ROS: observation, policy call and action processing of the faive_lab tasks
orca-v2-inhand-repose-v0 (proprioception only, 74 observations) and orca-v2-inhand-repose-extero-v0 (with the cube pose,
82 observations), shared by policy_inference_cube_spin_v2.py and eval_policies.py.

The policy is the TorchScript file that faive_lab's play.py exports (it includes the observation normalisation).
Each step:
1. Build the observation as in the simulation (USD joint order):
       joint_pos       joint positions normalised to [-1, 1] over the joint limits
       joint_vel       joint velocities * 0.2
       cube_pose       position (3), orientation w, x, y, z (4), tracked flag (1); only with use_cube_pose (CubePose)
       goal_command    [0, 0, -0.05, 0, 0, angular velocity about z]
       last_action     the policy's previous raw output
       last_joint_cmd  the previous joint position command (radians)
2. Run the policy, map its output onto the joint limits and smooth it with the same moving average as the
   simulation (target = alpha * new + (1 - alpha) * previous).

The joint limits, observation order and scales below must match the training task (orca_v2_repose_env_cfg.py).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

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
CUBE_POSE_DIM = 3 + 4 + 1
OBS_DIM_EXTERO = OBS_DIM + CUBE_POSE_DIM
CUBE_POSE_TIMEOUT_S = 0.25  # an older cube pose counts as lost; = timeout_s of CUBE_TRACKER in faive_lab

# hand_root in the frame the simulation observes the cube in (mdp/cube_tracker.py in faive_lab): robot init_state pos
# (0.0004, 0.1512, -0.0187), rot (w, x, y, z) (-0.2607, 0.9654, 0, 0) of the root link right_forearm, and
# hand_root -> forearm 90 deg about x (orcahand_v2.urdf.xacro): 120.2 deg about x.
OBS_T_HAND = np.array([
    [1.0,  0.0,     0.0,     0.0004],
    [0.0, -0.5034, -0.8641,  0.1512],
    [0.0,  0.8641, -0.5034, -0.0187],
    [0.0,  0.0,     0.0,     1.0],
])
# the mount in hand_root (orcahand_v2.urdf.xacro); the tag's pose is measured from it
MOUNT_IN_HAND = np.array([-0.01, 0.0, -0.058])


def observation_size(policy):
    """The number of observations a loaded TorchScript policy takes: OBS_DIM, OBS_DIM_EXTERO or None (neither)."""
    for size in (OBS_DIM, OBS_DIM_EXTERO):
        try:
            with torch.no_grad():
                policy(torch.zeros(1, size))
            return size
        except RuntimeError:
            continue
    return None


def hand_T_camera(cam_T_tag, hand_to_tag):
    """The camera's pose in hand_root (4x4) from the AprilTag calibration (cam_T_tag, apriltag_calibration.py) and the
    tag's pose measured from the mount (x, y, z in m; roll, pitch, yaw in deg), as cube_spin_v2.launch.py builds TF."""
    x, y, z, roll, pitch, yaw = hand_to_tag
    hand_T_tag = np.eye(4)
    hand_T_tag[:3, :3] = Rotation.from_euler("XYZ", [roll, pitch, yaw], degrees=True).as_matrix()
    hand_T_tag[:3, 3] = np.array([x, y, z]) + MOUNT_IN_HAND
    return hand_T_tag @ np.linalg.inv(np.asarray(cam_T_tag, dtype=float))


class CubePose:
    """The cube-pose part of the observation, as the simulated tracker gives it: the newest pose in the simulation's
    frame with the flag 1, or, while the cube is lost (no pose for timeout_s), the last pose with the flag 0."""

    def __init__(self, timeout_s=CUBE_POSE_TIMEOUT_S):
        self.timeout_s = float(timeout_s)
        self.pos = self.quat = self.stamp = None

    def update(self, stamp, hand_T_cube):
        """A new pose of the cube in hand_root (4x4) seen at stamp (seconds)."""
        if self.stamp is not None and stamp < self.stamp:
            return
        obs_T_cube = OBS_T_HAND @ np.asarray(hand_T_cube, dtype=float)
        x, y, z, w = Rotation.from_matrix(obs_T_cube[:3, :3]).as_quat()
        quat = np.array([w, x, y, z], dtype=np.float32)
        # q and -q are the same rotation; the simulation's quaternion changes continuously, so keep the previous sign
        if self.quat is not None and np.dot(quat, self.quat) < 0:
            quat = -quat
        self.pos, self.quat, self.stamp = obs_T_cube[:3, 3].astype(np.float32), quat, float(stamp)

    def age(self, now):
        return float("inf") if self.stamp is None else now - self.stamp

    def observation(self, now):
        """The CUBE_POSE_DIM values at time now, or None before the first pose."""
        if self.pos is None:
            return None
        tracked = 1.0 if self.age(now) <= self.timeout_s else 0.0
        return np.concatenate([self.pos, self.quat, [tracked]]).astype(np.float32)


class CubeSpinPolicy:
    """Joint positions and velocities in, joint position targets out, all in the hand driver's joint order (radians).

    use_cube_pose: True for a policy of the extero task (step() then needs the cube pose), None = what the policy takes.
    """

    def __init__(self, policy_path, joint_order, goal_ang_vel=1.0, action_alpha=0.2, velocity_obs_scale=0.2,
                 use_cube_pose=False):
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
        if use_cube_pose is None:
            use_cube_pose = observation_size(self.policy) == OBS_DIM_EXTERO
        self.use_cube_pose = bool(use_cube_pose)
        self.obs_dim = OBS_DIM_EXTERO if self.use_cube_pose else OBS_DIM
        with torch.no_grad():
            try:
                out = self.policy(torch.zeros(1, self.obs_dim))
            except RuntimeError as e:
                raise ValueError(f"{policy_path} does not take {self.obs_dim} observations (use_cube_pose="
                                 f"{self.use_cube_pose}; {OBS_DIM} without the cube pose, {OBS_DIM_EXTERO} with it): "
                                 f"{e}") from e
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

    def step(self, joint_pos, joint_vel, cube_pose=None):
        """One policy step; cube_pose is CubePose.observation() (only with use_cube_pose). Returns (target in the hand
        driver's order, observation, raw action, target in USD order), or None if the joint state is not finite or
        the policy needs the cube pose and there is none yet (nothing changes then)."""
        joint_pos_usd = np.asarray(joint_pos, dtype=np.float32)[self.robot_to_usd_idx]
        joint_vel_usd = np.asarray(joint_vel, dtype=np.float32)[self.robot_to_usd_idx]
        if not (np.all(np.isfinite(joint_pos_usd)) and np.all(np.isfinite(joint_vel_usd))):
            return None
        cube_obs = []
        if self.use_cube_pose:
            if cube_pose is None:
                return None
            cube_obs = [np.asarray(cube_pose, dtype=np.float32)]
        if self.last_joint_cmd_usd is None:
            self.last_joint_cmd_usd = joint_pos_usd.copy()

        obs = np.concatenate([
            self.normalize(joint_pos_usd),
            joint_vel_usd * self.velocity_obs_scale,
            *cube_obs,
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
