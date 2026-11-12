"""The digital twin in the evaluation's video (picture in picture): the hand model in its measured pose and the cube where
a policy with the cube pose sees it, in the simulation's frame, drawn offscreen by pybullet's CPU renderer in a process
of its own.

twin.Twin sends its state to state_q: (time, joint angles in the hand driver's order, cube position and quaternion
w x y z or None, cube tracked, status lines). The process puts (time, BGR panel) on image_q, and the camera process
pastes the newest panel into its frame (hud.picture_in_picture).
"""

import contextlib
import os
import queue
import re
import shutil
import signal
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

PANEL_SIZE = (300, 380)  # width, height (pixels)
RATE_HZ = 20.0
EYE, TARGET, FOV_DEG = (0.24, -0.2, 0.2), (0.0, 0.02, -0.035), 45.0  # in sim_world, as twin.py
LIGHT = dict(lightDirection=[0.3, -1.0, 1.5], lightAmbientCoeff=0.45, lightDiffuseCoeff=0.55, lightSpecularCoeff=0.05)
HAND_RGBA = [0.72, 0.74, 0.78, 1.0]
BACKGROUND = (48, 48, 48)
CUBE_DAE = "models/dexcube/dex_cube_instanceable.dae"
CUBE_TEXTURE = "models/dexcube/dex_cube_instanceable_texture_0001.png"


@contextlib.contextmanager
def quiet():
    """pybullet prints its build time and URDF warnings from C; keep them out of the evaluation's terminal."""
    fd = os.dup(1)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(fd, 1)
            os.close(fd)


def hand_urdf_for_pybullet(share, folder):
    """The URDF with each mesh as a file of its own in folder: pybullet does not load one mesh file for two links,
    nor a URDF in one long line."""
    from xml.dom import minidom
    from .twin import hand_urdf
    count = 0

    def copy(match):
        nonlocal count
        count += 1
        target = Path(folder) / f"{count}_{Path(match.group(1)).name}"
        shutil.copy(Path(share) / match.group(1), target)
        return str(target)

    path = Path(folder) / "orcahand_v2.urdf"
    path.write_text(minidom.parseString(re.sub(r"package://viz/([^\"']+)", copy, hand_urdf())).toprettyxml())
    return path


def cube_mesh(dae_path):
    """Vertices (+-1), indices, UVs and normals of the cube's COLLADA mesh, as the simulation and RViz show it."""
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    root = ET.parse(dae_path).getroot()
    arrays = {a.get("id"): np.array(a.text.split(), dtype=float) for a in root.iter(f"{{{ns['c']}}}float_array")}
    pick = lambda suffix: next(v for k, v in arrays.items() if k.endswith(suffix))  # noqa: E731
    indices = [int(i) for i in root.find(".//c:polylist/c:p", ns).text.split()]
    return (pick("positions-array").reshape(-1, 3), indices, pick("tex0-array").reshape(-1, 2),
            pick("normals-array").reshape(-1, 3))


def run(state_q, image_q, stop, joint_ids, cube_edge):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the main process stops this one
    try:
        _run(state_q, image_q, stop, list(joint_ids), float(cube_edge))
    except Exception:
        print("The digital twin in the video stopped:")
        traceback.print_exc()


def _run(state_q, image_q, stop, joint_ids, cube_edge):
    from ament_index_python.packages import get_package_share_directory
    from cube_spin_policy import GOAL_POS, OBS_T_HAND
    from .hud import GREEN, RED, WHITE, text

    share = Path(get_package_share_directory("viz"))
    with tempfile.TemporaryDirectory() as folder, quiet():
        import pybullet as p
        p.connect(p.DIRECT)
        hand = p.loadURDF(str(hand_urdf_for_pybullet(share, folder)), useFixedBase=True)
        x, y, z, w = Rotation.from_matrix(OBS_T_HAND[:3, :3]).as_quat()
        p.resetBasePositionAndOrientation(hand, OBS_T_HAND[:3, 3].tolist(), [x, y, z, w])
        index = {p.getJointInfo(hand, i)[1].decode(): i for i in range(p.getNumJoints(hand))}
        joints = [(index[name], k) for k, name in enumerate(joint_ids) if name in index]
        for link in range(-1, p.getNumJoints(hand)):
            p.changeVisualShape(hand, link, rgbaColor=HAND_RGBA)

        vertices, indices, uvs, normals = cube_mesh(share / CUBE_DAE)
        shape = p.createVisualShape(p.GEOM_MESH, vertices=(vertices * cube_edge / 2).tolist(), indices=indices,
                                    uvs=uvs.tolist(), normals=normals.tolist())
        cube = p.createMultiBody(baseVisualShapeIndex=shape, basePosition=[0, 0, -10])
        p.changeVisualShape(cube, -1, textureUniqueId=p.loadTexture(str(share / CUBE_TEXTURE)))
        lost = p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[cube_edge / 2] * 3, rgbaColor=[1.0, 0.25, 0.25, 1.0]), basePosition=[0, 0, -10])
        p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.004,
                                                                   rgbaColor=[0.2, 0.9, 0.3, 1.0]),
                          basePosition=list(GOAL_POS))

    width, height = PANEL_SIZE
    view = p.computeViewMatrix(EYE, TARGET, [0, 0, 1])
    projection = p.computeProjectionMatrixFOV(FOV_DEG, width / height, 0.01, 2.0)
    state, shown = None, None
    while not stop.is_set():
        t0 = time.time()
        try:
            while True:
                state = state_q.get_nowait()
        except queue.Empty:
            pass
        if state is not None and state is not shown:
            shown = state
            _, q, pos, quat, tracked, lines = state
            for j, k in joints:
                p.resetJointState(hand, j, float(q[k]))
            for body, visible in ((cube, pos is not None and tracked), (lost, pos is not None and not tracked)):
                if visible:
                    w_, x_, y_, z_ = quat
                    p.resetBasePositionAndOrientation(body, list(pos), [x_, y_, z_, w_])
                else:
                    p.resetBasePositionAndOrientation(body, [0, 0, -10], [0, 0, 0, 1])
            _, _, rgba, _, mask = p.getCameraImage(width, height, view, projection, renderer=p.ER_TINY_RENDERER,
                                                   shadow=0, **LIGHT)
            panel = cv2.cvtColor(np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            panel[np.reshape(mask, (height, width)) < 0] = BACKGROUND
            text(panel, "digital twin: what the policy sees", (8, 18), WHITE, 0.45)
            for i, line in enumerate(lines):
                color = GREEN if line == "cube tracked" else RED if line.startswith("cube lost") else WHITE
                text(panel, line, (8, height - 12 - 18 * (len(lines) - 1 - i)), color, 0.42)
            cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (200, 200, 200), 1)
            try:
                image_q.get_nowait()
            except queue.Empty:
                pass
            try:
                image_q.put_nowait((time.time(), panel))
            except queue.Full:
                pass
        time.sleep(max(0.0, 1.0 / RATE_HZ - (time.time() - t0)))
