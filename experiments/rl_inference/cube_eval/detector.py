"""Cube pose from one camera image: the Mask R-CNN keypoint detector of pose_estimation and PnP, without ROS.

The same steps as cube_detection_node.py: the network finds the cube's 8 corners, RANSAC PnP rejects wrong corners,
and an iterative refinement starts from the last pose. Runs on CUDA, on Apple's GPU (MPS) or on the CPU.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE.parent / "pose_estimation/finetuned_models/maskrcnn-finetuned_lightning_new_adamw.ckpt"


def cube_corners(edge):
    """The cube's corners in its own frame, in the detector's keypoint order."""
    h = edge / 2
    return np.array([[h, h, h], [-h, h, h], [h, -h, h], [-h, -h, h],
                     [h, h, -h], [-h, h, -h], [h, -h, -h], [-h, -h, -h]], dtype=np.float32)


def pick_device(name="auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CubePoseEstimator:
    def __init__(self, ckpt=DEFAULT_CKPT, cube_edge=0.045, device="auto", min_score=0.3, crop=None):
        # Apple Silicon: Mask R-CNN's nms has no MPS implementation, run it on the CPU. Before importing torch.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        sys.path.insert(0, str(HERE.parent))
        from pose_estimation.maskrcnn_detection import maskrcnn_detection

        if not Path(ckpt).exists():
            raise FileNotFoundError(f"{ckpt} not found: run experiments/rl_inference/pose_estimation/download_models.sh")
        self.torch = torch
        self.device = pick_device(device)
        self.model = maskrcnn_detection.load_from_checkpoint(str(ckpt), map_location="cpu").to(self.device).eval()
        self.model_points = cube_corners(cube_edge)
        self.min_score = min_score
        self.crop = crop  # (x0, y0, x1, y1) in pixels: only this part of the image goes to the network
        self.last_pose = None  # (rvec, tvec) of the last good detection, the start of the refinement
        self.misses = 0

    def detect(self, image_bgr, K, dist=None):
        """(R 3x3, t 3, corner pixels 8x2, score) of the cube in the camera frame, or None."""
        torch = self.torch
        x0, y0 = 0, 0
        if self.crop is not None:
            x0, y0, x1, y1 = self.crop
            image_bgr = image_bgr[y0:y1, x0:x1]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).to(self.device)
        with torch.no_grad():
            output = self.model.get_detections([tensor])[0]
        scores = output["scores"].detach().cpu().numpy()
        if len(scores) == 0 or scores.max() < self.min_score:
            return self._miss()
        best = int(scores.argmax())
        corners = output["keypoints"][best].detach().cpu().numpy()[:, :2].astype(np.float32) + [x0, y0]
        corners = corners.astype(np.float32)
        K = np.asarray(K, dtype=np.float32)
        dist = None if dist is None else np.asarray(dist, dtype=np.float32)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(self.model_points, corners, K, dist, iterationsCount=500,
                                                     reprojectionError=3.0, confidence=0.99, flags=cv2.SOLVEPNP_AP3P)
        if not ok or inliers is None or len(inliers) < 4:
            return self._miss()
        idx = inliers.ravel()
        guess_r, guess_t = self.last_pose if self.last_pose is not None else (rvec, tvec)
        ok, rvec2, tvec2 = cv2.solvePnP(self.model_points[idx], corners[idx], K, dist, rvec=guess_r.copy(),
                                        tvec=guess_t.copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok and np.linalg.norm(tvec2 - tvec) < 0.1:  # a refinement that ran away from RANSAC's pose is not used
            rvec, tvec = rvec2, tvec2
        distance = float(np.linalg.norm(tvec))
        if not 0.05 < distance < 2.0:
            return self._miss()
        self.last_pose, self.misses = (rvec.copy(), tvec.copy()), 0
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.ravel().astype(float), corners, float(scores[best])

    def _miss(self):
        self.misses += 1
        if self.misses >= 5:
            self.last_pose = None
        return None
