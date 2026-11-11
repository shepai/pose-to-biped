"""
Procedural generator of balanced, collision-free whole-body reference motions for any robot in
G1_tele.robots (SyntheticMotionGenerator(robot="h1"), or pass a Kinematics).

Why: a pose-tracking policy needs lots of varied *feasible* targets for hands, feet, elbows and
knees. Random robot poses solved with whole-body IK give exactly that, with the matching joint
angles for free (used for reward shaping and reference-state initialisation).

Every clip is quasi-static: the centre of mass is kept over the support foot/feet by an IK task,
transitions are minimum-jerk and slow enough that a real robot could follow them. Clip kinds:

    upper    feet planted at a fixed stance, small pelvis motion, arms + waist move a lot
    whole    feet planted, squats / leans / twists / weight shifts + arms + waist
    balance  as `whole` plus lifting one foot (knee raises, foot reaches) and putting it back
    step     as `balance` but the lifted foot is placed somewhere new (stepping, turning)

Other robots: the numbers below were tuned on the G1 and are converted to the robot's size with
Kinematics.to_height / to_length / to_half_width (heights bend the legs as much, steps and lifts
scale with leg length, stances with hip width). Upper-body poses are drawn from the robot's
RobotSpec.explore ranges (written for the left side, mirrored for the right), and robots without
an ankle roll joint keep the soles level about the foot's length only (the roll is left free).
"""
import numpy as np

from G1_tele.motion.kinematics import Kinematics, quat_from_rpy, quat_to_mat, rot_z
from G1_tele.robots.spec import ARM_PARTS, HEAD_PARTS, SIDES, WAIST_PARTS

UPPER = slice(12, 29)          # the G1's upper body (other robots: Kinematics.upper_idx)


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def foot_tasks(k, L, R, w_pos=200.0, w_rot=50.0):
    """IK tasks putting both feet flat on the floor: L, R = (x, y, lift, yaw) of each ankle. Robots
    without an ankle roll joint only keep the sole level about the foot's width (pitch) and its yaw:
    with no joint to roll the foot, insisting on it would fight the leg."""
    axis_w = None if k.has_ankle_roll else (0.05, 1.0, 1.0)
    out = []
    for side, (body, f) in enumerate(zip(k.foot_ids, (L, R))):
        out.append({"type": "point", "body": body, "offset": np.zeros(3),
                    "target": np.array([f[0], f[1], k.ankle_height + f[2]]), "w": w_pos})
        rt = {"type": "orient", "body": body, "target": k.foot_flat(side, f[3]), "w": w_rot}
        if axis_w is not None:
            rt["axes"], rt["axis_w"] = rot_z(f[3]), axis_w
        out.append(rt)
    return out


def random_upper(k, rng, amount=1.0):
    """Random waist / arm (/ head) joint angles for the joints in k.upper_idx, from the robot's
    RobotSpec.explore ranges; locked joints and joints without a range stay at their default.
    amount in [0, 1] scales the spread (0 = default pose). Draws are in a fixed order (waist yaw, roll,
    pitch, then each arm's parts), so the G1 gets the same poses for the same seed as before."""
    spec, r, a = k.spec, rng, amount
    q0 = k.default_q.astype(float)
    up = list(k.upper_idx)
    q = q0[up].copy()
    pos = {spec.joints[i].slot: n for n, i in enumerate(up)}
    free = {spec.joints[i].slot for i in up if not spec.joints[i].locked}
    ex = spec.explore
    for part in WAIST_PARTS:
        if part in pos and part in free and part in ex:
            q[pos[part]] = r.uniform(*ex[part]) * a + (1 - a) * q0[up[pos[part]]]
    arm_parts = [p for p in ARM_PARTS if p in ex]
    sym = r.random() < 0.3
    for side in SIDES:
        slots = {p: f"{side}_{p}" for p in arm_parts}
        if side == "right" and sym:                          # mirror the left arm
            for p, sl in slots.items():
                if sl in pos and sl in free and f"left_{p}" in pos:
                    q[pos[sl]] = spec.mirror_sign(p) * q[pos[f"left_{p}"]]
            continue
        if r.random() < 0.15:                              # keep this arm near rest
            continue
        sgn = lambda p: 1.0 if side == "left" else spec.mirror_sign(p)       # noqa: E731
        for p, sl in slots.items():
            u = r.uniform(*ex[p])                          # drawn even for missing joints: same stream
            if sl in pos and sl in free:
                il = spec.index(f"left_{p}")
                rest = q0[il] if il is not None else sgn(p) * q0[up[pos[sl]]]
                q[pos[sl]] = sgn(p) * (u * a + (1 - a) * rest)
    for part in HEAD_PARTS:
        if part in pos and part in free and part in ex:
            q[pos[part]] = r.uniform(*ex[part]) * a + (1 - a) * q0[up[pos[part]]]
    return q


class SyntheticMotionGenerator:
    def __init__(self, kin=None, fps=50, seed=0, collision_check_every=4, max_joint_vel=3.0, verbose=True,
                 robot=None):
        self.k = kin or Kinematics(robot)
        self.fps = fps
        self.max_joint_vel = max_joint_vel
        self.rng = np.random.default_rng(seed)
        self.check_every = collision_check_every
        self.verbose = verbose
        k = self.k
        self.fl, self.fr = k.foot_ids
        self.upper = np.array(k.upper_idx, int)
        legs = k.v_adr[k.leg_idx]
        # IK may move the floating base and the legs; waist/arms are set directly
        self.active = np.zeros(k.nv, bool)
        self.active[:6] = True
        self.active[legs] = True
        pw = np.zeros(k.nv)
        pw[legs] = 2e-3
        yaws = [k.spec.index(f"{s}_hip_yaw") for s in SIDES]
        pw[[k.v_adr[i] for i in yaws if i is not None]] = 2e-2
        self.posture_w = pw
        self.posture = k.default_qpos()

    # ------------------------------------------------------------------ random parameters
    def _upper(self, amount=1.0):
        """Random waist + arm joint angles (wrists stay put). amount in [0,1] scales the spread."""
        return random_upper(self.k, self.rng, amount)

    def _foot_center(self, foot):
        x, y, z, yaw = foot
        return np.array([x, y]) + rot_z(yaw)[:2, :2] @ self.k.foot_center

    def _initial_state(self, kind):
        r, k = self.rng, self.k
        half = k.to_half_width(r.uniform(0.10, 0.17))
        yaw_l, yaw_r = r.uniform(-0.05, 0.3), -r.uniform(-0.05, 0.3)
        dx = k.to_length(r.uniform(-0.08, 0.08))
        x0 = k.stance_x
        return {
            "L": np.array([x0 + dx, half, 0.0, yaw_l]), "R": np.array([x0 - dx, -half, 0.0, yaw_r]),
            "w": 0.5, "h": k.to_height(r.uniform(0.70, 0.78)), "rpy": np.zeros(3),
            "upper": self._upper(0.5 * r.random()),
        }

    def _pose_target(self, s, kind):
        """New double-support pose from state s (feet unchanged)."""
        r, k = self.rng, self.k
        H = k.to_height
        t = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in s.items()}
        if kind == "upper":
            t["h"] = np.clip(s["h"] + k.to_length(r.uniform(-0.03, 0.03)), H(0.70), H(0.79))
            t["rpy"] = np.array([r.uniform(-0.04, 0.04), r.uniform(-0.05, 0.1), r.uniform(-0.15, 0.15)])
            t["w"] = r.uniform(0.4, 0.6)
        else:
            t["h"] = r.choice([H(r.uniform(0.70, 0.79)), H(r.uniform(0.58, 0.79))])
            t["rpy"] = np.array([r.uniform(-0.12, 0.12), r.uniform(-0.1, 0.4), r.uniform(-0.35, 0.35)])
            t["w"] = r.uniform(0.2, 0.8)
        t["upper"] = self._upper(r.uniform(0.3, 1.0))
        return t

    # ------------------------------------------------------------------ IK for one frame
    def _solve(self, q_prev, s, iters=6):
        k = self.k
        L, R = s["L"], s["R"]
        q = q_prev.copy()
        q[k.q_adr[self.upper]] = s["upper"]
        com = s["w"] * self._foot_center(L) + (1 - s["w"]) * self._foot_center(R)
        yaw_mid = 0.5 * (L[3] + R[3])
        roll, pitch, dyaw = s["rpy"]
        tasks = foot_tasks(k, L, R) + [
            {"type": "com", "target": com, "w": 100.0},
            {"type": "root_height", "target": s["h"], "w": 10.0},
            {"type": "root_orient", "target": quat_to_mat(quat_from_rpy(roll, pitch, yaw_mid + dyaw)), "w": 3.0},
        ]
        post = self.posture.copy()
        post[:7] = q[:7]
        q, _ = k.solve_ik(q, tasks, active=self.active, posture=post, posture_w=self.posture_w, iters=iters)
        # accuracy check
        k.forward(q)
        fe = max(np.linalg.norm(k.d.xpos[self.fl] - tasks[0]["target"]), np.linalg.norm(k.d.xpos[self.fr] - tasks[2]["target"]))
        ce = np.linalg.norm(k.com(q)[:2] - com)
        return q, fe, ce

    # ------------------------------------------------------------------ segments
    def _segment(self, q, s0, s1, dur, swing=None, lift=0.0, _retry=True):
        """Interpolate s0 -> s1 over dur seconds, solving IK each frame. Returns frames or None.
        If any joint would move faster than self.max_joint_vel the segment is slowed down."""
        frames, q_end = self._segment_once(q, s0, s1, dur, swing, lift)
        if frames is not None and _retry:
            v = np.max(np.abs(np.diff(np.array([q] + frames)[:, self.k.q_adr], axis=0))) * self.fps
            if v > self.max_joint_vel:
                return self._segment(q, s0, s1, dur * 1.1 * v / self.max_joint_vel, swing, lift, _retry=False)
        return frames, q_end

    def _segment_once(self, q, s0, s1, dur, swing=None, lift=0.0):
        n = max(2, int(round(dur * self.fps)))
        frames = []
        for i in range(1, n + 1):
            a = min_jerk(i / n)
            s = {key: (1 - a) * s0[key] + a * s1[key] for key in ("L", "R", "w", "h", "rpy", "upper")}
            if swing is not None and lift > 0:              # foot travels on an arc
                s[swing] = s[swing].copy()
                s[swing][2] += lift * np.sin(np.pi * i / n)
            q, fe, ce = self._solve(q, s)
            if fe > 0.01 or ce > 0.02:
                return None, q
            if self.check_every and (i % self.check_every == 0 or i == n):
                if self.k.self_collisions(q):
                    return None, q
            frames.append(q.copy())
        return frames, q

    def _hold(self, q, dur):
        return [q.copy() for _ in range(int(round(dur * self.fps)))]

    def _move_dur(self, s0, s1, lo=0.8):
        dj = np.max(np.abs(s1["upper"] - s0["upper"]))
        dh = abs(s1["h"] - s0["h"]) / self.k.scale
        dw = abs(s1["w"] - s0["w"])
        dr = np.max(np.abs(s1["rpy"] - s0["rpy"]))
        base = max(0.55 * dj, 6.0 * dh, 1.6 * dw, 2.0 * dr)
        return float(np.clip(lo + base * self.rng.uniform(0.8, 1.4), lo, 3.5))

    def _single_leg(self, q, s, place_new):
        """Shift weight, lift a foot (hold / reach), put it back or step. Returns (frames, q, s)."""
        r, k = self.rng, self.k
        swing = "L" if r.random() < 0.5 else "R"
        stance = "R" if swing == "L" else "L"
        out = []
        # 1. weight onto stance foot
        s1 = dict(s); s1["w"] = 1.0 if stance == "L" else 0.0
        s1["rpy"] = np.array([0.0, r.uniform(-0.05, 0.15), s["rpy"][2] * 0.5])
        s1["h"] = np.clip(s["h"], k.to_height(0.68), k.to_height(0.78))
        seg, q = self._segment(q, s, s1, r.uniform(0.9, 1.4))
        if seg is None:
            return None, q, s
        out += seg
        st = s1[stance]
        sgn = 1.0 if swing == "L" else -1.0
        if place_new:                                       # 2a. step to a new spot
            R = rot_z(st[3])
            fwd = k.to_length(r.uniform(-0.18, 0.25))
            lat = 2 * k.to_half_width(r.uniform(0.17, 0.34) / 2)
            dyaw = sgn * r.uniform(-0.15, 0.45)
            xy = st[:2] + R[:2, :2] @ np.array([fwd, sgn * lat])
            target = np.array([xy[0], xy[1], 0.0, st[3] + dyaw])
            s2 = dict(s1); s2[swing] = target
            s2["upper"] = self._upper(r.uniform(0.0, 0.6)) if r.random() < 0.5 else s1["upper"]
            seg, q = self._segment(q, s1, s2, r.uniform(0.8, 1.2), swing=swing, lift=k.to_length(r.uniform(0.05, 0.10)))
            if seg is None:
                return None, q, s
            out += seg
            s3 = dict(s2)
        else:                                               # 2b. lift and hold, then put back
            home = s1[swing].copy()
            R = rot_z(home[3])
            off = R[:2, :2] @ np.array([k.to_length(r.uniform(-0.12, 0.25)), sgn * k.to_length(r.uniform(-0.02, 0.12))])
            lifted = home.copy(); lifted[:2] += off; lifted[2] = k.to_length(r.uniform(0.05, 0.25))
            s2 = dict(s1); s2[swing] = lifted
            s2["upper"] = self._upper(r.uniform(0.2, 0.9))
            seg, q = self._segment(q, s1, s2, r.uniform(0.9, 1.5))
            if seg is None:
                return None, q, s
            out += seg + self._hold(q, r.uniform(0.3, 1.5))
            s3 = dict(s2); s3[swing] = home
            seg, q = self._segment(q, s2, s3, r.uniform(0.9, 1.3))
            if seg is None:
                return None, q, s
            out += seg
        # 3. weight back to both feet
        s4 = dict(s3); s4["w"] = r.uniform(0.35, 0.65)
        seg, q = self._segment(q, s3, s4, r.uniform(0.8, 1.2))
        if seg is None:
            return None, q, s
        out += seg
        return out, q, s4

    # ------------------------------------------------------------------ clips
    def clip(self, kind="whole", seconds=20.0, max_tries=400):
        r, k = self.rng, self.k
        for _ in range(50):                                 # find a valid start pose
            s = self._initial_state(kind)
            q0 = k.default_qpos()
            q0[2] = s["h"]
            q, fe, ce = q0, 1, 1
            for _ in range(6):
                q, fe, ce = self._solve(q, s, iters=30)
            if fe < 0.005 and ce < 0.01 and not k.self_collisions(q):
                break
        frames = self._hold(q, r.uniform(0.3, 1.0))
        tries = 0
        while len(frames) < seconds * self.fps and tries < max_tries:
            tries += 1
            u = r.random()
            if kind in ("balance", "step") and u < (0.45 if kind == "step" else 0.35):
                seg, q2, s2 = self._single_leg(q, s, place_new=(kind == "step" and r.random() < 0.75))
            else:
                t = self._pose_target(s, kind)
                seg, q2 = self._segment(q, s, t, self._move_dur(s, t))
                s2 = t
                if seg is not None:
                    seg += self._hold(q2, r.choice([0.0, r.uniform(0.2, 1.2)]))
            if seg is None:
                continue                                    # infeasible: discard, keep old state
            frames += seg
            q, s = q2, s2
        Q = np.array(frames)
        return Q[:, :7].copy(), Q[:, k.q_adr].copy()

    def library(self, kinds=("upper", "whole", "balance", "step"), clips_per_kind=25, seconds=20.0):
        clips, names, kk = [], [], []
        for kind in kinds:
            for i in range(clips_per_kind):
                clips.append(self.clip(kind, seconds))
                names.append(f"{kind}_{i:03d}"); kk.append(kind)
                if self.verbose:
                    print(f"  {kind:8s} clip {i + 1}/{clips_per_kind}: {len(clips[-1][0]) / self.fps:.1f} s", flush=True)
        return clips, names, kk
