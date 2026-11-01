"""
Procedural generator of balanced, collision-free whole-body reference motions for the G1.

Why: a pose-tracking policy needs lots of varied *feasible* targets for hands, feet, elbows and
knees. Random robot poses solved with whole-body IK give exactly that, with the matching joint
angles for free (used for reward shaping and reference-state initialisation).

Every clip is quasi-static: the centre of mass is kept over the support foot/feet by an IK task,
transitions are minimum-jerk and slow enough that a real robot could follow them. Clip kinds:

    upper    feet planted at a fixed stance, small pelvis motion, arms + waist move a lot
    whole    feet planted, squats / leans / twists / weight shifts + arms + waist
    balance  as `whole` plus lifting one foot (knee raises, foot reaches) and putting it back
    step     as `balance` but the lifted foot is placed somewhere new (stepping, turning)
"""
import numpy as np

from G1_tele.envs.whole_body import BODY_JOINTS, DEFAULT_Q, LEG_SLICE
from G1_tele.motion.kinematics import (ANKLE_HEIGHT, FOOT_CENTER, G1Kinematics, quat_from_rpy, quat_to_mat,
                                       rot_z, yaw_of_quat)

UPPER = slice(12, 29)
J = {n: i for i, n in enumerate(BODY_JOINTS)}


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


class SyntheticMotionGenerator:
    def __init__(self, kin=None, fps=50, seed=0, collision_check_every=4, max_joint_vel=3.0, verbose=True):
        self.k = kin or G1Kinematics()
        self.fps = fps
        self.max_joint_vel = max_joint_vel
        self.rng = np.random.default_rng(seed)
        self.check_every = collision_check_every
        self.verbose = verbose
        k = self.k
        self.fl, self.fr = k.foot_ids
        # IK may move the floating base and the legs; waist/arms are set directly
        self.active = np.zeros(k.nv, bool)
        self.active[:6] = True
        self.active[k.v_adr[LEG_SLICE]] = True
        pw = np.zeros(k.nv)
        pw[k.v_adr[LEG_SLICE]] = 2e-3
        pw[[k.v_adr[J["left_hip_yaw_joint"]], k.v_adr[J["right_hip_yaw_joint"]]]] = 2e-2
        self.posture_w = pw
        self.posture = k.default_qpos()

    # ------------------------------------------------------------------ random parameters
    def _upper(self, amount=1.0):
        """Random waist + arm joint angles (wrists stay 0). amount in [0,1] scales the spread."""
        r = self.rng
        q = DEFAULT_Q[UPPER].astype(float).copy()
        a = amount
        q[0] = r.uniform(-0.6, 0.6) * a                    # waist yaw
        q[1] = r.uniform(-0.2, 0.2) * a                    # waist roll
        q[2] = r.uniform(-0.1, 0.35) * a                   # waist pitch (+ = lean forward)
        sym = r.random() < 0.3
        for side, base in (("left", 3), ("right", 10)):
            sgn = 1.0 if side == "left" else -1.0
            if side == "right" and sym:
                q[10:14] = q[3:7] * np.array([1, -1, -1, 1]); continue
            if r.random() < 0.15:                          # keep this arm near rest
                continue
            q[base + 0] = r.uniform(-2.6, 0.7) * a + (1 - a) * q[base + 0]   # shoulder pitch (- = raise forward)
            q[base + 1] = sgn * (r.uniform(-0.15, 1.7) * a + (1 - a) * 0.15)  # shoulder roll (outward)
            q[base + 2] = r.uniform(-0.9, 0.9) * a                         # shoulder yaw
            q[base + 3] = r.uniform(-0.7, 1.55) * a + (1 - a) * 0.3       # elbow (1.57 = straight)
        return q

    def _foot_center(self, foot):
        x, y, z, yaw = foot
        return np.array([x, y]) + rot_z(yaw)[:2, :2] @ FOOT_CENTER

    def _initial_state(self, kind):
        r = self.rng
        half = r.uniform(0.10, 0.17)
        yaw_l, yaw_r = r.uniform(-0.05, 0.3), -r.uniform(-0.05, 0.3)
        dx = r.uniform(-0.08, 0.08)
        return {
            "L": np.array([-0.026 + dx, half, 0.0, yaw_l]), "R": np.array([-0.026 - dx, -half, 0.0, yaw_r]),
            "w": 0.5, "h": r.uniform(0.70, 0.78), "rpy": np.zeros(3), "upper": self._upper(0.5 * r.random()),
        }

    def _pose_target(self, s, kind):
        """New double-support pose from state s (feet unchanged)."""
        r = self.rng
        t = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in s.items()}
        if kind == "upper":
            t["h"] = np.clip(s["h"] + r.uniform(-0.03, 0.03), 0.70, 0.79)
            t["rpy"] = np.array([r.uniform(-0.04, 0.04), r.uniform(-0.05, 0.1), r.uniform(-0.15, 0.15)])
            t["w"] = r.uniform(0.4, 0.6)
        else:
            t["h"] = r.choice([r.uniform(0.70, 0.79), r.uniform(0.58, 0.79)])
            t["rpy"] = np.array([r.uniform(-0.12, 0.12), r.uniform(-0.1, 0.4), r.uniform(-0.35, 0.35)])
            t["w"] = r.uniform(0.2, 0.8)
        t["upper"] = self._upper(r.uniform(0.3, 1.0))
        return t

    # ------------------------------------------------------------------ IK for one frame
    def _solve(self, q_prev, s, iters=6):
        k = self.k
        L, R = s["L"], s["R"]
        q = q_prev.copy()
        q[k.q_adr[UPPER]] = s["upper"]
        com = s["w"] * self._foot_center(L) + (1 - s["w"]) * self._foot_center(R)
        yaw_mid = 0.5 * (L[3] + R[3])
        roll, pitch, dyaw = s["rpy"]
        tasks = [
            {"type": "point", "body": self.fl, "offset": np.zeros(3), "target": np.array([L[0], L[1], ANKLE_HEIGHT + L[2]]), "w": 200.0},
            {"type": "orient", "body": self.fl, "target": rot_z(L[3]), "w": 50.0},
            {"type": "point", "body": self.fr, "offset": np.zeros(3), "target": np.array([R[0], R[1], ANKLE_HEIGHT + R[2]]), "w": 200.0},
            {"type": "orient", "body": self.fr, "target": rot_z(R[3]), "w": 50.0},
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
        dh = abs(s1["h"] - s0["h"])
        dw = abs(s1["w"] - s0["w"])
        dr = np.max(np.abs(s1["rpy"] - s0["rpy"]))
        base = max(0.55 * dj, 6.0 * dh, 1.6 * dw, 2.0 * dr)
        return float(np.clip(lo + base * self.rng.uniform(0.8, 1.4), lo, 3.5))

    def _single_leg(self, q, s, place_new):
        """Shift weight, lift a foot (hold / reach), put it back or step. Returns (frames, q, s)."""
        r = self.rng
        swing = "L" if r.random() < 0.5 else "R"
        stance = "R" if swing == "L" else "L"
        out = []
        # 1. weight onto stance foot
        s1 = dict(s); s1["w"] = 1.0 if stance == "L" else 0.0
        s1["rpy"] = np.array([0.0, r.uniform(-0.05, 0.15), s["rpy"][2] * 0.5])
        s1["h"] = np.clip(s["h"], 0.68, 0.78)
        seg, q = self._segment(q, s, s1, r.uniform(0.9, 1.4))
        if seg is None:
            return None, q, s
        out += seg
        st = s1[stance]
        sgn = 1.0 if swing == "L" else -1.0
        if place_new:                                       # 2a. step to a new spot
            R = rot_z(st[3])
            fwd = r.uniform(-0.18, 0.25)
            lat = r.uniform(0.17, 0.34)
            dyaw = sgn * r.uniform(-0.15, 0.45)
            xy = st[:2] + R[:2, :2] @ np.array([fwd, sgn * lat])
            target = np.array([xy[0], xy[1], 0.0, st[3] + dyaw])
            s2 = dict(s1); s2[swing] = target
            s2["upper"] = self._upper(r.uniform(0.0, 0.6)) if r.random() < 0.5 else s1["upper"]
            seg, q = self._segment(q, s1, s2, r.uniform(0.8, 1.2), swing=swing, lift=r.uniform(0.05, 0.10))
            if seg is None:
                return None, q, s
            out += seg
            s3 = dict(s2)
        else:                                               # 2b. lift and hold, then put back
            home = s1[swing].copy()
            R = rot_z(home[3])
            off = R[:2, :2] @ np.array([r.uniform(-0.12, 0.25), sgn * r.uniform(-0.02, 0.12)])
            lifted = home.copy(); lifted[:2] += off; lifted[2] = r.uniform(0.05, 0.25)
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
