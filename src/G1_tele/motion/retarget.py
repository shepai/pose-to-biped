"""
Keypoints -> robot joint angles (whole-body IK retargeting).

Use it for
  * turning recorded keypoint clips (e.g. from a body tracker) into a motion library for training,
  * online, at deployment time, to turn live tracker keypoints into the reference joint angles the
    pose-tracking policy observes (see KeypointRetargeter.step).

Input keypoints are in the heading frame of G1_tele.motion.kinematics (x forward, y left, z up,
origin = pelvis xy on the floor), already scaled to robot size. Any subset of KEYPOINTS works;
missing ones are filled in by a posture prior (the default pose), so 4 points (hands + feet) are
enough, and elbows / knees / hips / shoulders make the solution better defined.
"""
import numpy as np

from G1_tele.envs.whole_body import BODY_JOINTS, WRIST_IDX
from G1_tele.motion.kinematics import ANKLE_HEIGHT, KEYPOINTS, G1Kinematics, rot_z, yaw_of_quat

DEFAULT_WEIGHTS = {"hand": 10.0, "foot": 20.0, "toe": 5.0, "elbow": 3.0, "knee": 3.0,
                   "shoulder": 3.0, "hip": 5.0, "head": 2.0}


def _weight(name, weights):
    for key, w in weights.items():
        if key in name:
            return w
    return 1.0


class KeypointRetargeter:
    """Stateful per-frame IK (warm-started from the previous frame)."""

    def __init__(self, kin=None, weights=None, contact_height=0.06, com_weight=0.0, posture_weight=2e-3,
                 lock_wrists=True):
        self.k = kin or G1Kinematics()
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.contact_height = contact_height
        self.com_weight = com_weight
        k = self.k
        self.active = np.ones(k.nv, bool)
        if lock_wrists:
            self.active[k.v_adr[WRIST_IDX]] = False
        pw = np.full(k.nv, posture_weight)
        pw[:3] = 0.0                                        # root translation is set by keypoints
        pw[3:6] = 10 * posture_weight                       # prefer an upright pelvis
        self.posture_w = pw
        self.reset()

    def reset(self):
        self.q = self.k.default_qpos()
        self.first = True

    def step(self, keypoints):
        """keypoints: dict name -> (3,) heading-frame position. Returns (root (7,), joints (29,), err)."""
        k = self.k
        tasks = [{"type": "root_xy", "target": np.zeros(2), "w": 200.0},
                 {"type": "root_yaw", "target": 0.0, "w": 100.0}]
        feet_down = []
        for name, p in keypoints.items():
            body, off = KEYPOINTS[name]
            p = np.array(p, float)
            w = _weight(name, self.weights)
            if name in ("left_foot", "right_foot") and p[2] < ANKLE_HEIGHT + self.contact_height:
                p[2] = max(p[2], ANKLE_HEIGHT)
                bid = k.m.body(body).id
                k.forward(self.q)
                yaw = np.arctan2(k.d.xmat[bid].reshape(3, 3)[1, 0], k.d.xmat[bid].reshape(3, 3)[0, 0])
                tasks.append({"type": "orient", "body": bid, "target": rot_z(yaw), "w": 10.0})
                feet_down.append(p[:2])
            tasks.append({"type": "point", "body": k.m.body(body).id, "offset": np.array(off, float),
                          "target": p, "w": w})
        if self.com_weight > 0 and feet_down:
            tasks.append({"type": "com", "target": np.mean(feet_down, 0) + [0.03, 0.0], "w": self.com_weight})
        post = k.default_qpos()
        post[:3] = self.q[:3]
        if not self.first:                                  # temporal smoothness: blend towards last frame
            post[7:] = 0.5 * post[7:] + 0.5 * self.q[7:]
        q, err = k.solve_ik(self.q, tasks, active=self.active, posture=post, posture_w=self.posture_w,
                            iters=60 if self.first else 12)
        self.q, self.first = q, False
        names = list(keypoints)
        got = k.keypoints_local(names, q)
        kp_err = float(np.mean(np.linalg.norm(got - np.array([keypoints[n] for n in names]), axis=1)))
        return q[:7].copy(), q[k.q_adr].copy(), kp_err


def retarget_clip(keypoints, kin=None, **kw):
    """keypoints: dict name -> (T,3) arrays. Returns root (T,7), joints (T,29), mean keypoint error (T,)."""
    rt = KeypointRetargeter(kin, **kw)
    T = len(next(iter(keypoints.values())))
    roots, joints, errs = [], [], []
    for t in range(T):
        r, j, e = rt.step({n: v[t] for n, v in keypoints.items()})
        roots.append(r); joints.append(j); errs.append(e)
    return np.array(roots), np.array(joints), np.array(errs)


def hands_relative_to_shoulders(hands, kin=None, human_arm_len=0.65, robot_arm_len=0.50, max_reach=0.52):
    """Convert the old (T,2,3) 'hand relative to its shoulder, body frame' data into heading-frame
    keypoints for hands, with feet and hips pinned at the default stance."""
    k = kin or G1Kinematics()
    q0 = k.default_qpos()
    q0[2] -= k.keypoints_local(["left_foot"], q0)[0, 2] - ANKLE_HEIGHT       # feet on the floor
    names = ["left_shoulder", "right_shoulder", "left_foot", "right_foot", "left_hip", "right_hip"]
    base = dict(zip(names, k.keypoints_local(names, q0)))
    h = np.asarray(hands, float).reshape(-1, 2, 3) * (robot_arm_len / human_arm_len)
    n = np.linalg.norm(h, axis=-1, keepdims=True)
    h = h * np.minimum(1.0, max_reach / np.maximum(n, 1e-6))
    T = len(h)
    out = {"left_hand": base["left_shoulder"] + h[:, 0], "right_hand": base["right_shoulder"] + h[:, 1]}
    for nm in ("left_foot", "right_foot", "left_hip", "right_hip"):
        out[nm] = np.repeat(base[nm][None], T, 0)
    return out


def resample(keypoints, src_fps, dst_fps):
    T = len(next(iter(keypoints.values())))
    t_src = np.arange(T) / src_fps
    t_dst = np.arange(0, t_src[-1] + 1e-9, 1.0 / dst_fps)
    return {n: np.stack([np.interp(t_dst, t_src, v[:, i]) for i in range(3)], 1) for n, v in keypoints.items()}
