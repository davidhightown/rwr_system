"""The cube-spin policy without ROS: observation, policy call and action processing of the faive_lab tasks
orca-v2-inhand-repose-v0 (proprioception only, 74 observations) and orca-v2-inhand-repose-extero-v0 (with the cube pose,
82 observations), shared by policy_inference_cube_spin_v2.py and eval_policies.py, and of the other policies on W&B
that eval_policies.py runs with --mode (LAYOUTS).

The policy is the TorchScript file that faive_lab's play.py exports (it includes the observation normalisation).
Each step:
1. Build the observation as in the simulation (USD joint order):
       joint_pos         joint positions normalised to [-1, 1] over the joint limits
       joint_vel         joint velocities * 0.2
       cube_pose         position (3), orientation w, x, y, z (4), tracked flag (1); only with use_cube_pose (CubePose)
       goal_command      [0, 0, -0.05, angular velocity about x, y, z]; about z unless axis says otherwise
       goal_orientation  instead of goal_command in the goal task: goal orientation (4) and the rotation from the goal
                         to the tracked orientation (4) (GoalOrientations)
       last_action       the policy's previous raw output
       last_joint_cmd    the previous joint position command (radians)
   A term with a history has its last n values, oldest first, as Isaac Lab's per-term history (flatten_history_dim);
   after a reset the first value fills the history.
2. Run the policy, map its output onto the joint limits and smooth it with the same moving average as the
   simulation (target = alpha * new + (1 - alpha) * previous).

The joint limits, observation order and scales below must match the training task (orca_v2_repose_env_cfg.py).
"""

from collections import deque
from typing import NamedTuple

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
GOAL_POS = (0.0, 0.0, -0.05)  # where the task commands the cube, in the frame the simulation observes it in
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}  # of the frame the simulation observes in
GOAL_TOLERANCE_RAD = 0.4  # a goal orientation counts as reached (DeXtreme's test tolerance; 0.1 in training)

TERM_SIZES = dict(joint_pos=NUM_JOINTS, joint_vel=NUM_JOINTS, cube_pose=CUBE_POSE_DIM, goal_command=6,
                  goal_orientation=8, last_action=NUM_JOINTS, last_joint_cmd=NUM_JOINTS)
SPIN = ("joint_pos", "joint_vel", "goal_command", "last_action", "last_joint_cmd")
SPIN_EXTERO = ("joint_pos", "joint_vel", "cube_pose", "goal_command", "last_action", "last_joint_cmd")
GOALS = ("joint_pos", "joint_vel", "cube_pose", "goal_orientation", "last_action", "last_joint_cmd")


class Layout(NamedTuple):
    terms: tuple   # in the order of the observation
    history: dict  # term -> number of steps, for the terms with a history


# the observation layouts by size; the size of a policy file tells its layout
LAYOUTS = {
    OBS_DIM: Layout(SPIN, {}),
    OBS_DIM_EXTERO: Layout(SPIN_EXTERO, {}),
    84: Layout(GOALS, {}),  # orca-v2-inhand-goal-v0 (manta)
    210: Layout(SPIN, {"joint_pos": 5, "last_joint_cmd": 5}),  # orca-v2-inhand-repose-adr-v0 (newt, tapir)
    740: Layout(SPIN, dict.fromkeys(SPIN, 10)),  # students with a history (cygnet_hist, lark_hist)
}
assert all(sum(TERM_SIZES[t] * l.history.get(t, 1) for t in l.terms) == size for size, l in LAYOUTS.items())

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
RECURRENT_STATE = ("hidden_state", "cell_state")  # buffers of an exported recurrent policy (rsl_rl's exporter)


def input_size(policy):
    """The number of observations a loaded TorchScript policy takes: a key of LAYOUTS if it takes one of them, else
    the size of its normalisation or first layer, or None if that is not to be found."""
    for size in LAYOUTS:
        try:
            with torch.no_grad():
                policy(torch.zeros(1, size))
            return size
        except RuntimeError:
            continue
    buffers = dict(policy.named_buffers())
    if "normalizer._mean" in buffers:
        return int(buffers["normalizer._mean"].shape[-1])
    return next((int(p.shape[-1]) for _, p in policy.named_parameters() if p.dim() == 2), None)


def is_recurrent(policy):
    return any(name in RECURRENT_STATE for name, _ in policy.named_buffers())


def reset_recurrent(policy):
    """Clear the memory of a recurrent policy (its export keeps it in buffers and has no reset)."""
    for name, buffer in policy.named_buffers():
        if name in RECURRENT_STATE:
            buffer.zero_()


def hand_T_camera(cam_T_tag, hand_to_tag):
    """The camera's pose in hand_root (4x4) from the AprilTag calibration (cam_T_tag, apriltag_calibration.py) and the
    tag's pose measured from the mount (x, y, z in m; roll, pitch, yaw in deg), as cube_spin_v2.launch.py builds TF."""
    x, y, z, roll, pitch, yaw = hand_to_tag
    hand_T_tag = np.eye(4)
    hand_T_tag[:3, :3] = Rotation.from_euler("XYZ", [roll, pitch, yaw], degrees=True).as_matrix()
    hand_T_tag[:3, 3] = np.array([x, y, z]) + MOUNT_IN_HAND
    return hand_T_tag @ np.linalg.inv(np.asarray(cam_T_tag, dtype=float))


def axis_in_camera(hand_T_camera, axis):
    """An axis of the frame the simulation observes the cube in (x, y or z), in the camera frame."""
    obs_T_camera = OBS_T_HAND @ np.asarray(hand_T_camera, dtype=float)
    return obs_T_camera[:3, :3].T @ np.array(AXES[axis])


def wxyz(rotation):
    x, y, z, w = rotation.as_quat()
    return np.array([w, x, y, z])


def rotation_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w])


def unique(quat_wxyz):
    """q or -q, the one with w >= 0 (Isaac Lab's quat_unique)."""
    return -quat_wxyz if quat_wxyz[0] < 0 else quat_wxyz


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
        quat = wxyz(Rotation.from_matrix(obs_T_cube[:3, :3])).astype(np.float32)
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


class GoalOrientations:
    """The goals of the goal task (orca-v2-inhand-goal-v0, after DeXtreme): uniformly random orientations of the cube in
    the frame the simulation observes it in, the next one as soon as the tracked orientation is within ``tolerance``
    (rad) of the goal. The observation (8) is the goal (w, x, y, z) and the rotation from the goal to the tracked
    orientation, tracked * conj(goal), both with w >= 0, as cube_pose_tracked_goal in faive_lab computes them."""

    def __init__(self, tolerance=GOAL_TOLERANCE_RAD, seed=None):
        self.tolerance = float(tolerance)
        self.rng = np.random.default_rng(seed)
        self.reached = 0    # goals reached since the start (a reset does not change it)
        self.error = None   # rad from the tracked orientation to the goal at the last step
        self.quat = None
        self.new_goal()

    def new_goal(self):
        q = self.rng.normal(size=4)  # a normalised 4D Gaussian is uniform over the rotations
        self.quat = unique(q / np.linalg.norm(q)).astype(np.float32)

    def _to_goal(self, cube_quat):
        rotation = rotation_of(cube_quat) * rotation_of(self.quat).inv()
        self.error = float(rotation.magnitude())
        return unique(wxyz(rotation)).astype(np.float32)

    def observation(self, cube_pose):
        """The 8 values for a CubePose.observation(); first the next goal if the tracked cube reached this one."""
        to_goal = self._to_goal(cube_pose[3:7])
        if cube_pose[7] > 0.5 and self.error < self.tolerance:
            self.reached += 1
            self.new_goal()
            to_goal = self._to_goal(cube_pose[3:7])
        return np.concatenate([self.quat, to_goal]).astype(np.float32)


class CubeSpinPolicy:
    """Joint positions and velocities in, joint position targets out, all in the hand driver's joint order (radians).

    use_cube_pose: True for a policy of the extero task, False for one of the proprioceptive task (the two layouts of
    the ROS node), None = whatever layout of LAYOUTS the policy takes (step() then needs the cube pose if the layout
    has it). axis: the axis of the goal angular velocity (all-axes task). A policy of the goal task gets the goal
    orientations of ``goals`` (GoalOrientations with goal_tolerance, default GOAL_TOLERANCE_RAD, and seed). A recurrent
    policy's memory is cleared at every reset().
    """

    def __init__(self, policy_path, joint_order, goal_ang_vel=1.0, action_alpha=0.2, velocity_obs_scale=0.2,
                 use_cube_pose=False, axis="z", goal_tolerance=None, seed=None):
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
        self.goal_command = np.array([*GOAL_POS, *(float(goal_ang_vel) * np.array(AXES[axis]))], dtype=np.float32)

        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        size = input_size(self.policy)
        if use_cube_pose is not None and size != (OBS_DIM_EXTERO if use_cube_pose else OBS_DIM):
            raise ValueError(f"{policy_path} does not take {OBS_DIM_EXTERO if use_cube_pose else OBS_DIM} observations "
                             f"(use_cube_pose={use_cube_pose}; {OBS_DIM} without the cube pose, {OBS_DIM_EXTERO} with "
                             f"it) but {size}")
        if size not in LAYOUTS:
            raise ValueError(f"{policy_path} takes {size} observations; the known layouts have {sorted(LAYOUTS)}")
        self.obs_dim, self.layout = size, LAYOUTS[size]
        self.use_cube_pose = "cube_pose" in self.layout.terms
        self.recurrent = is_recurrent(self.policy)
        self.goals = None
        if "goal_orientation" in self.layout.terms:
            self.goals = GoalOrientations(GOAL_TOLERANCE_RAD if goal_tolerance is None else goal_tolerance, seed)
        with torch.no_grad():
            out = self.policy(torch.zeros(1, self.obs_dim))
        if out.shape[-1] != NUM_JOINTS:
            raise ValueError(f"{policy_path} outputs {out.shape[-1]} actions, expected {NUM_JOINTS}")
        self.reset()

    def reset(self):
        """As at a reset in the task: last action zero, histories and memory empty, a new goal orientation; the moving
        average starts from the pose of the next step."""
        self.last_action_usd = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.last_joint_cmd_usd = None
        self.history = {}
        if self.recurrent:
            reset_recurrent(self.policy)
        if self.goals is not None:
            self.goals.new_goal()

    def flip_direction(self):
        self.goal_command[3:] *= -1.0

    def normalize(self, joint_pos_usd):
        """Isaac Lab's scale_transform (joint_pos_limit_normalized): limits -> [-1, 1], not clipped."""
        low, high = self.limits_usd[:, 0], self.limits_usd[:, 1]
        return (2.0 * (joint_pos_usd - (low + high) / 2) / (high - low)).astype(np.float32)

    def denormalize(self, action_usd):
        """Isaac Lab's unscale_transform (rescale_to_limits): [-1, 1] -> limits."""
        low, high = self.limits_usd[:, 0], self.limits_usd[:, 1]
        return (low + (action_usd + 1.0) / 2 * (high - low)).astype(np.float32)

    def _with_history(self, term, value):
        steps = self.layout.history.get(term, 1)
        if steps <= 1:
            return value
        if term not in self.history:
            self.history[term] = deque([value] * steps, maxlen=steps)
        else:
            self.history[term].append(value)
        return np.concatenate(self.history[term])

    def step(self, joint_pos, joint_vel, cube_pose=None):
        """One policy step; cube_pose is CubePose.observation() (only with use_cube_pose). Returns (target in the hand
        driver's order, observation, raw action, target in USD order), or None if the joint state is not finite or
        the policy needs the cube pose and there is none yet (nothing changes then)."""
        joint_pos_usd = np.asarray(joint_pos, dtype=np.float32)[self.robot_to_usd_idx]
        joint_vel_usd = np.asarray(joint_vel, dtype=np.float32)[self.robot_to_usd_idx]
        if not (np.all(np.isfinite(joint_pos_usd)) and np.all(np.isfinite(joint_vel_usd))):
            return None
        if self.use_cube_pose and cube_pose is None:
            return None
        if self.last_joint_cmd_usd is None:
            self.last_joint_cmd_usd = joint_pos_usd.copy()

        values = dict(joint_pos=self.normalize(joint_pos_usd), joint_vel=joint_vel_usd * self.velocity_obs_scale,
                      goal_command=self.goal_command.copy(), last_action=self.last_action_usd,
                      last_joint_cmd=self.last_joint_cmd_usd)
        if self.use_cube_pose:
            values["cube_pose"] = np.asarray(cube_pose, dtype=np.float32)
        if self.goals is not None:
            values["goal_orientation"] = self.goals.observation(values["cube_pose"])
        obs = np.concatenate([self._with_history(t, values[t]) for t in self.layout.terms]).astype(np.float32)

        with torch.no_grad():
            action_usd = self.policy(torch.from_numpy(obs).unsqueeze(0))[0].numpy().astype(np.float32)
        self.last_action_usd = action_usd

        # as the task's EMAJointPositionToLimitsAction
        target_usd = self.denormalize(np.clip(action_usd, -1.0, 1.0))
        target_usd = self.action_alpha * target_usd + (1.0 - self.action_alpha) * self.last_joint_cmd_usd
        target_usd = np.clip(target_usd, self.limits_usd[:, 0], self.limits_usd[:, 1]).astype(np.float32)
        self.last_joint_cmd_usd = target_usd
        return target_usd[self.usd_to_robot_idx], obs, action_usd, target_usd
