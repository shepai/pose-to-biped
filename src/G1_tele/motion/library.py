"""
Motion library: a set of reference clips stored as full robot configurations.

File format (.npz)
------------------
    root     (N, 7)  float32   pelvis position (world, m) + quaternion (w, x, y, z)
    joints   (N, 29) float32   the 29 body joints in G1_tele.envs.whole_body.BODY_JOINTS order (rad)
    starts   (C+1,)  int       clip c is rows starts[c]:starts[c+1]
    fps      ()      float     frame rate of every clip
    names    (C,)    str       optional clip names
    kinds    (C,)    str       optional clip category, used for sampling weights ("upper", "whole", ...)

Keypoints are not stored: the env computes them by forward kinematics, so the same file works
whatever keypoint set you train with.
"""
from pathlib import Path

import numpy as np

from G1_tele.envs.whole_body import BODY_JOINTS
from G1_tele.motion.kinematics import slerp

MOTION_DIR = Path(__file__).resolve().parent.parent / "assets" / "motions"
DEFAULT_MOTIONS = MOTION_DIR / "synthetic.npz"


def save_library(path, clips, fps, names=None, kinds=None):
    """clips: list of (root (T,7), joints (T,29)) pairs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    starts = np.cumsum([0] + [len(r) for r, _ in clips])
    np.savez_compressed(
        path,
        root=np.concatenate([r for r, _ in clips]).astype(np.float32),
        joints=np.concatenate([j for _, j in clips]).astype(np.float32),
        starts=starts, fps=np.float32(fps),
        names=np.array(names if names is not None else [f"clip_{i}" for i in range(len(clips))]),
        kinds=np.array(kinds if kinds is not None else ["clip"] * len(clips)),
        joint_names=np.array(BODY_JOINTS))


def merge_libraries(out_path, *paths):
    clips, names, kinds, fps = [], [], [], None
    for p in paths:
        lib = MotionLibrary(p)
        if fps is not None and abs(lib.fps - fps) > 1e-6:
            raise ValueError("all libraries must share the same fps")
        fps = lib.fps
        for c in range(lib.n_clips):
            clips.append((lib.root[lib.starts[c]:lib.starts[c + 1]], lib.joints[lib.starts[c]:lib.starts[c + 1]]))
        names += list(lib.names); kinds += list(lib.kinds)
    save_library(out_path, clips, fps, names, kinds)


class MotionLibrary:
    def __init__(self, paths):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        roots, joints, starts, names, kinds, fps = [], [], [0], [], [], None
        for p in paths:
            f = np.load(p, allow_pickle=False)
            if "joint_names" in f and list(f["joint_names"]) != BODY_JOINTS:
                raise ValueError(f"{p}: joint order differs from whole_body.BODY_JOINTS")
            if fps is not None and abs(float(f["fps"]) - fps) > 1e-6:
                raise ValueError("all motion files must share the same fps")
            fps = float(f["fps"])
            off = starts[-1]
            roots.append(f["root"].astype(np.float64)); joints.append(f["joints"].astype(np.float64))
            starts += list(off + f["starts"][1:])
            n = len(f["starts"]) - 1
            names += list(f["names"]) if "names" in f else [f"{Path(p).stem}_{i}" for i in range(n)]
            kinds += list(f["kinds"]) if "kinds" in f else ["clip"] * n
        self.root = np.concatenate(roots)
        self.root[:, 3:7] /= np.linalg.norm(self.root[:, 3:7], axis=1, keepdims=True)
        self.joints = np.concatenate(joints)
        self.starts = np.array(starts)
        self.names, self.kinds = np.array(names), np.array(kinds)
        self.fps = fps
        self.n_clips = len(self.starts) - 1
        self.lengths = np.diff(self.starts)

    def duration(self, c):
        return (self.lengths[c] - 1) / self.fps

    def sample_clip(self, rng, kind_weights=None, min_seconds=1.0):
        w = np.array([self.duration(c) for c in range(self.n_clips)])
        w[w < min_seconds] = 0.0
        if kind_weights:
            w *= np.array([kind_weights.get(k, 0.0) for k in self.kinds])
        if w.sum() <= 0:
            raise ValueError("no clip matches kind_weights / min_seconds")
        return int(rng.choice(self.n_clips, p=w / w.sum()))

    def _idx(self, c, t):
        x = np.clip(t * self.fps, 0.0, self.lengths[c] - 1)
        k = int(np.floor(x))
        k2 = min(k + 1, self.lengths[c] - 1)
        return self.starts[c] + k, self.starts[c] + k2, x - k

    def frame(self, c, t):
        """Interpolated (root (7,), joints (29,)) at time t (clamped to the clip)."""
        i, j, f = self._idx(c, t)
        root = np.empty(7)
        root[:3] = (1 - f) * self.root[i, :3] + f * self.root[j, :3]
        root[3:] = slerp(self.root[i, 3:], self.root[j, 3:], f)
        return root, (1 - f) * self.joints[i] + f * self.joints[j]

    def velocity(self, c, t):
        """(root linear vel world (3,), root angular vel body frame (3,), joint vel (29,)) at t."""
        h = 1.0 / self.fps
        t0 = min(max(t, 0.0), self.duration(c) - h) if self.duration(c) > h else 0.0
        r0, j0 = self.frame(c, t0)
        r1, j1 = self.frame(c, t0 + h)
        lin = (r1[:3] - r0[:3]) / h
        q0, q1 = r0[3:], r1[3:]
        # body-frame angular velocity: 2 * (q0^-1 * q1).vec / h
        w0, v0 = q0[0], -q0[1:]
        w1, v1 = q1[0], q1[1:]
        dw = w0 * w1 - v0 @ v1
        dv = w0 * v1 + w1 * v0 + np.cross(v0, v1)
        if dw < 0:
            dw, dv = -dw, -dv
        ang = 2.0 * dv / h
        return lin, ang, (j1 - j0) / h
