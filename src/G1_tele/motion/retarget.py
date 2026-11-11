"""
Keypoints -> robot joint angles (whole-body IK retargeting).

Use it for
  * turning recorded keypoint clips (e.g. from a body tracker) into a motion library for training,
  * online, at deployment time, to turn live tracker keypoints into the reference joint angles the
    pose-tracking policy observes (see KeypointRetargeter.step).

Input keypoints are in the heading frame of G1_tele.motion.kinematics (x forward, y left, z up,
origin = base xy on the floor), already scaled to robot size. Any subset of the robot's keypoints works;
missing ones are filled in by a posture prior (the default pose), so 4 points (hands + feet) are
enough, and elbows / knees / hips / shoulders make the solution better defined.
"""
import numpy as np

from G1_tele.motion.kinematics import Kinematics, rot_z

DEFAULT_WEIGHTS = {"hand": 10.0, "foot": 20.0, "toe": 5.0, "elbow": 3.0, "knee": 3.0,
                   "shoulder": 3.0, "hip": 5.0, "head": 2.0}


def _weight(name, weights):
    for key, w in weights.items():
        if key in name:
            return w
    return 1.0


FLAT_FADE = 0.04       # m above contact_height over which the flat-foot hold fades out


class KeypointRetargeter:
    """Stateful per-frame IK (warm-started from the previous frame)."""

    def __init__(self, kin=None, weights=None, contact_height=0.06, com_weight=0.0, posture_weight=2e-3,
                 lock_wrists=True, robot=None):
        self.k = kin or Kinematics(robot)
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        k = self.k
        self.contact_height = contact_height * k.scale
        self.com_weight = com_weight
        self.active = np.ones(k.nv, bool)
        if lock_wrists:                                     # the robot's locked joints (wrists, head) stay put
            self.active[k.v_adr[k.locked_idx]] = False
        pw = np.full(k.nv, posture_weight)
        pw[:3] = 0.0                                        # root translation is set by keypoints
        pw[3:6] = 10 * posture_weight                       # prefer an upright pelvis
        self.posture_w = pw
        self.reset()

    def reset(self):
        self.q = self.k.default_qpos()
        self.first = True

    def step(self, keypoints):
        """keypoints: dict name -> (3,) heading-frame position. Returns (root (7,), joints (J,), err)."""
        k = self.k
        A, fade_h = k.ankle_height, FLAT_FADE * k.scale
        tasks = [{"type": "root_xy", "target": np.zeros(2), "w": 200.0},
                 {"type": "root_yaw", "target": 0.0, "w": 100.0}]
        feet_down = []
        for name, p in keypoints.items():
            body, off = k.keypoints[name]
            p = np.array(p, float)
            w = _weight(name, self.weights)
            if name in ("left_foot", "right_foot") and p[2] < A + self.contact_height + fade_h:
                # a foot near the floor is held flat; the hold fades out over FLAT_FADE above contact_height,
                # so the ankle doesn't snap to a new angle the moment the foot crosses that height
                fade = np.clip((A + self.contact_height + fade_h - p[2]) / fade_h, 0.0, 1.0)
                side = 0 if name == "left_foot" else 1
                bid = k.m.body(body).id
                k.forward(self.q)
                yaw = k.foot_yaw(side, k.d.xmat[bid].reshape(3, 3))
                t = {"type": "orient", "body": bid, "target": k.foot_flat(side, yaw), "w": 10.0 * fade}
                if not k.has_ankle_roll:                    # no joint can roll the foot: keep it level lengthwise
                    t["axes"], t["axis_w"] = rot_z(yaw), (0.05, 1.0, 1.0)
                tasks.append(t)
                if p[2] < A + self.contact_height:
                    p[2] = max(p[2], A)
                    feet_down.append(p[:2])
            tasks.append({"type": "point", "body": k.m.body(body).id, "offset": np.array(off, float),
                          "target": p, "w": w})
        if self.com_weight > 0 and feet_down:
            tasks.append({"type": "com", "target": np.mean(feet_down, 0) + [0.03 * k.scale, 0.0], "w": self.com_weight})
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
    """keypoints: dict name -> (T,3) arrays. Returns root (T,7), joints (T,J), mean keypoint error (T,)."""
    rt = KeypointRetargeter(kin, **kw)
    T = len(next(iter(keypoints.values())))
    roots, joints, errs = [], [], []
    for t in range(T):
        r, j, e = rt.step({n: v[t] for n, v in keypoints.items()})
        roots.append(r); joints.append(j); errs.append(e)
    return np.array(roots), np.array(joints), np.array(errs)


def hands_relative_to_shoulders(hands, kin=None, human_arm_len=0.65, robot_arm_len=None, max_reach=None):
    """Convert the old (T,2,3) 'hand relative to its shoulder, body frame' data into heading-frame
    keypoints for hands, with feet and hips pinned at the default stance. The G1 numbers (arm 0.50 m,
    reach 0.52 m) are scaled to other robots by arm length."""
    k = kin or Kinematics()
    arm = k.geometry["arm_length"] / 0.4487                 # the G1's shoulder -> elbow -> palm
    robot_arm_len = 0.50 * arm if robot_arm_len is None else robot_arm_len
    max_reach = 0.52 * arm if max_reach is None else max_reach
    q0 = k.default_qpos()
    q0[2] -= k.keypoints_local(["left_foot"], q0)[0, 2] - k.ankle_height       # feet on the floor
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
