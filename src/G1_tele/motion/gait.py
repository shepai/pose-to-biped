"""
Walking and turning reference clips for any robot in G1_tele.robots: clip kinds "turn", "walk" and "loco".
(Speeds, step lengths and heights were tuned on the G1 and are scaled to the robot's leg length and
hip width, like SyntheticMotionGenerator's.)

    turn   turning on the spot, left and right, with stops in between; arms near rest
    walk   walking forwards, backwards and sideways (gentle turns), starting and stopping; arms near rest
    loco   everything at once: walking, turning and side steps while the arms and waist move

How a clip is made
------------------
1. Command: a velocity (forward, sideways, turning rate) that changes every few seconds, with stops,
   smoothed by acceleration limits. Integrating it gives the path the body should follow.
2. Footsteps: the robot steps while the command is non-zero, or until its feet are square under the
   body again after a stop. Each foot lands where the body will be half a step after it lands, half a
   stance width to the side, and always within reach of the other foot (no crossed or colliding feet).
3. Balance: the zero-moment point (ZMP) stays on the stance foot during single support and moves to
   the next foot during double support. Preview control of the linear inverted pendulum (Kajita et al.
   2003) turns that ZMP plan into a centre-of-mass path, so the walk is dynamically balanced rather
   than just slow.
4. Whole-body IK puts the feet, the centre of mass and the pelvis where they were planned, frame by
   frame; the swing foot travels on a smooth arc, and the waist and arms are set directly.

Clips are 50 Hz like the rest of the motion library and are saved with save_library(); the root
moves, so these clips are what teaches the policy to follow root motion (PoseTrackingEnv with
locomotion=True).
"""
import numpy as np
from scipy.linalg import solve_discrete_are

from G1_tele.motion.kinematics import quat_from_rpy, quat_to_mat, rot_z
from G1_tele.motion.synthetic import SyntheticMotionGenerator, foot_tasks, min_jerk

LOCO_KINDS = ("turn", "walk", "loco")
G = 9.81


# ---------------------------------------------------------------------- ZMP preview control
def preview_gains(dt, zc, horizon, qe=1.0, r=1e-6):
    """Gains of the ZMP preview controller for the cart-table model (Katayama 1985, Kajita 2003).
    State x = [c, dc, ddc] per axis, input = jerk, output ZMP p = c - zc/g * ddc."""
    A = np.array([[1.0, dt, dt * dt / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    B = np.array([[dt ** 3 / 6], [dt * dt / 2], [dt]])
    C = np.array([[1.0, 0.0, -zc / G]])
    At = np.zeros((4, 4))
    At[0, 0] = 1.0
    At[0, 1:] = (C @ A)[0]
    At[1:, 1:] = A
    Bt = np.vstack([C @ B, B])
    It = np.array([[1.0], [0.0], [0.0], [0.0]])
    Q = np.zeros((4, 4))
    Q[0, 0] = qe
    P = solve_discrete_are(At, Bt, Q, np.array([[r]]))
    S = r + (Bt.T @ P @ Bt).item()
    K = (Bt.T @ P @ At) / S                               # [Gi, Gx]
    Gi, Gx = float(K[0, 0]), K[0, 1:].copy()
    Ac = At - Bt @ K
    X = -Ac.T @ P @ It
    Gp = np.empty(horizon)
    Gp[0] = -Gi
    for j in range(1, horizon):
        Gp[j] = (Bt.T @ X).item() / S
        X = Ac.T @ X
    return A, B[:, 0], C[0], Gi, Gx, Gp


def com_from_zmp(zmp, dt, zc, c0=None, preview_s=1.6):
    """Centre-of-mass path (n, 2) whose cart-table ZMP follows zmp (n, 2). Also returns that ZMP."""
    n = len(zmp)
    H = int(round(preview_s / dt))
    A, B, C, Gi, Gx, Gp = preview_gains(dt, zc, H)
    ref = np.vstack([zmp, np.repeat(zmp[-1:], H + 1, axis=0)])
    com, p_out = np.zeros((n, 2)), np.zeros((n, 2))
    for ax in range(2):
        x = np.array([zmp[0, ax] if c0 is None else c0[ax], 0.0, 0.0])
        e = 0.0
        for k in range(n):
            p = C @ x
            e += p - ref[k, ax]
            u = -Gi * e - Gx @ x - Gp @ ref[k + 1:k + 1 + H, ax]
            com[k, ax], p_out[k, ax] = x[0], p
            x = A @ x + B * u
    return com, p_out


# ---------------------------------------------------------------------- generator
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GaitGenerator:
    """Makes "turn", "walk" and "loco" clips. Shares the robot model, IK settings and random arm
    poses with SyntheticMotionGenerator (pass one in, a Kinematics via kin=, or robot="h1")."""

    def __init__(self, kin=None, fps=50, seed=0, synth=None, collision_check_every=4, max_joint_vel=10.0,
                 verbose=True, robot=None):
        self.s = synth or SyntheticMotionGenerator(kin, fps=fps, seed=seed, verbose=False, robot=robot)
        self.k = self.s.k
        self.fps = fps
        self.dt = 1.0 / fps
        self.rng = np.random.default_rng(seed + 7919)
        self.s.rng = self.rng                                   # random arm poses from the same stream
        self.check_every = collision_check_every
        self.max_joint_vel = max_joint_vel
        self.verbose = verbose
        self.fl, self.fr = self.k.foot_ids
        self.stats = {"clips": 0, "attempts": 0}

    # ------------------------------------------------------------------ commands
    def _command_segment(self, kind):
        r = self.rng
        if r.random() < (0.25 if kind != "loco" else 0.15):
            return np.zeros(3)                                  # stop
        if kind == "turn":
            return np.array([0.0, 0.0, r.choice([-1, 1]) * r.uniform(0.25, 0.6)])
        if kind == "walk":
            u = r.random()
            if u < 0.55:                                        # forwards (sometimes backwards)
                v = np.array([r.uniform(0.1, 0.4) if r.random() < 0.8 else -r.uniform(0.08, 0.15), 0.0, 0.0])
            elif u < 0.8:                                       # side steps
                v = np.array([0.0, r.choice([-1, 1]) * r.uniform(0.06, 0.14), 0.0])
            else:                                               # diagonal
                v = np.array([r.uniform(0.05, 0.25), r.choice([-1, 1]) * r.uniform(0.04, 0.1), 0.0])
            v[2] = r.uniform(-0.15, 0.15) if r.random() < 0.5 else 0.0
            return v
        # loco: anything
        u = r.random()
        if u < 0.3:
            return np.array([0.0, 0.0, r.choice([-1, 1]) * r.uniform(0.2, 0.6)])
        v = np.array([r.uniform(-0.12, 0.35), r.uniform(-0.1, 0.1) if r.random() < 0.5 else 0.0, 0.0])
        v[2] = r.uniform(-0.45, 0.45) if r.random() < 0.6 else 0.0
        return v

    def _commands(self, kind, n, start):
        """(n, 3) body-frame command: piecewise constant targets, acceleration limited."""
        r = self.rng
        target = np.zeros((n, 3))
        i = start
        while i < n:
            seg = int(r.uniform(2.0, 5.0) * self.fps)
            target[i:i + seg] = self._command_segment(kind)
            i += seg
        sc = np.array([self.k.scale, self.k.scale, 1.0])       # G1 speeds -> this robot's size
        target = target * sc
        acc = np.array([0.6, 0.4, 1.2]) * sc * self.dt         # m/s^2, m/s^2, rad/s^2 per step
        cmd = np.zeros((n, 3))
        for i in range(1, n):
            cmd[i] = cmd[i - 1] + np.clip(target[i] - cmd[i - 1], -acc, acc)
        cmd[np.abs(cmd) < 1e-4] = 0.0
        return cmd

    def _path(self, cmd):
        path = np.zeros((len(cmd), 3))
        for i in range(1, len(cmd)):
            x, y, yaw = path[i - 1]
            c, s = np.cos(yaw), np.sin(yaw)
            vx, vy, wz = cmd[i - 1]
            path[i] = [x + (c * vx - s * vy) * self.dt, y + (s * vx + c * vy) * self.dt, yaw + wz * self.dt]
        return path

    # ------------------------------------------------------------------ footsteps
    def _centre(self, foot):
        x, y, z, yaw = foot
        return np.array([x, y]) + rot_z(yaw)[:2, :2] @ self.k.foot_center

    def _nominal(self, pose, side, half_w):
        """Ankle pose (x, y, z, yaw) of a foot standing square under body pose (x, y, yaw)."""
        x, y, yaw = pose
        R = rot_z(yaw)[:2, :2]
        sgn = 1.0 if side == "L" else -1.0
        xy = np.array([x, y]) + R @ (np.array([0.0, sgn * half_w]) - self.k.foot_center)
        return np.array([xy[0], xy[1], 0.0, yaw])

    def _reach(self, target, stance, side):
        """Clamp a landing pose to what one step from the stance foot can reach without collisions."""
        k = self.k
        sgn = 1.0 if side == "L" else -1.0
        Rs = rot_z(stance[3])[:2, :2]
        rel = Rs.T @ (target[:2] - stance[:2])
        rel[0] = np.clip(rel[0], k.to_length(-0.18), k.to_length(0.28))
        rel[1] = sgn * np.clip(sgn * rel[1], 2 * k.to_half_width(0.085), 2 * k.to_half_width(0.18))
        dyaw = _wrap(target[3] - stance[3])
        dyaw = np.clip(dyaw, -0.15, 0.45) if sgn > 0 else np.clip(dyaw, -0.45, 0.15)   # toe-out more than toe-in
        xy = stance[:2] + Rs @ rel
        return np.array([xy[0], xy[1], 0.0, stance[3] + dyaw])

    def _plan(self, kind, seconds):
        """Feet (n, 4) each, ZMP (n, 2), commands and the gait parameters for one clip."""
        r, fps = self.rng, self.fps
        n = int(round(seconds * fps))
        hold = int(r.uniform(0.4, 1.0) * fps)
        k = self.k
        p = dict(half_w=k.to_half_width(r.uniform(0.105, 0.13)), t_ss=r.uniform(0.42, 0.55),
                 t_ds=r.uniform(0.10, 0.16), lift=k.to_length(r.uniform(0.05, 0.08)),
                 h=k.to_height(r.uniform(*k.spec.gait_height)))
        near = k.to_length(0.025)                               # "feet square under the body" tolerance
        cmd = self._commands(kind, n, hold)
        path = self._path(cmd)
        nss, nds = int(round(p["t_ss"] * fps)), int(round(p["t_ds"] * fps))
        feet = {s: self._nominal(path[0], s, p["half_w"]) for s in "LR"}
        F = {s: np.zeros((n, 4)) for s in "LR"}
        zmp = np.zeros((n, 2))
        mid = lambda: 0.5 * (self._centre(feet["L"]) + self._centre(feet["R"]))   # noqa: E731
        i, swing, standing = 0, None, True
        while i < n:
            ahead = min(n - 1, i + nds)
            moving = np.any(cmd[i] != 0) or np.any(cmd[ahead] != 0)
            if swing is None:                                   # first step: open towards the motion
                c = cmd[ahead]
                swing = "L" if (c[1] > 0.01 or c[2] > 0.05) else "R" if (c[1] < -0.01 or c[2] < -0.05) \
                    else ("L" if r.random() < 0.5 else "R")
            square = all(np.linalg.norm(feet[s][:2] - self._nominal(path[i], s, p["half_w"])[:2]) < near
                         and abs(_wrap(feet[s][3] - path[i, 2])) < 0.06 for s in "LR")
            if not moving and square:                           # stand still, weight in the middle
                F["L"][i], F["R"][i] = feet["L"], feet["R"]
                zmp[i] = mid() if standing else zmp[i - 1] + np.clip(mid() - zmp[i - 1], -0.01, 0.01)
                standing = standing or np.linalg.norm(zmp[i] - mid()) < 1e-6
                i += 1
                continue
            stance = "R" if swing == "L" else "L"
            n_ds = int(round(0.25 * fps)) if standing else nds  # the first weight shift is slower
            standing = False
            z0 = zmp[i - 1] if i > 0 else mid()
            zs = self._centre(feet[stance])
            for k in range(min(n_ds, n - i)):                  # double support: ZMP to the stance foot
                a = (k + 1) / n_ds
                F["L"][i + k], F["R"][i + k] = feet["L"], feet["R"]
                zmp[i + k] = (1 - a) * z0 + a * zs
            i += n_ds
            if i >= n:
                break
            t_ref = min(n - 1, i + nss + nds + nss // 2)       # body pose when this foot is next under it
            target = self._reach(self._nominal(path[t_ref], swing, p["half_w"]), feet[stance], swing)
            start = feet[swing].copy()
            for k in range(min(nss, n - i)):                   # single support: the swing foot's arc
                s = (k + 1) / nss
                a = min_jerk(s)
                f = start + a * (target - start)
                f[3] = start[3] + a * _wrap(target[3] - start[3])
                f[2] = p["lift"] * 16.0 * s * s * (1 - s) ** 2
                F[swing][i + k], F[stance][i + k] = f, feet[stance]
                zmp[i + k] = zs
            feet[swing] = target
            i += nss
            swing = stance
        return F, zmp, cmd, path, p

    # ------------------------------------------------------------------ arms and waist
    def _upper_track(self, kind, n):
        r = self.rng
        lo, hi = (0.0, 0.35) if kind in ("turn", "walk") else (0.25, 0.85)
        out = np.zeros((n, len(self.s.upper)))
        q0 = self.s._upper(r.uniform(lo, hi) * 0.5)
        i = 0
        while i < n:
            q1 = self.s._upper(r.uniform(lo, hi))
            m = int(r.uniform(1.0, 2.5) * self.fps)
            hold = int(r.uniform(0.0, 1.5) * self.fps)
            for k in range(m):
                if i + k < n:
                    out[i + k] = q0 + min_jerk((k + 1) / m) * (q1 - q0)
            out[i + m:i + m + hold] = q1
            i += m + hold
            q0 = q1
        return out

    # ------------------------------------------------------------------ IK
    def _solve(self, q, L, R, com, h, upper, pitch, iters):
        k = self.k
        q = q.copy()
        q[k.q_adr[self.s.upper]] = upper
        yaw = L[3] + 0.5 * _wrap(R[3] - L[3])
        tasks = foot_tasks(k, L, R) + [
            {"type": "com", "target": com, "w": 100.0},
            {"type": "root_height", "target": h, "w": 10.0},
            {"type": "root_orient", "target": quat_to_mat(quat_from_rpy(0.0, pitch, yaw)), "w": 3.0},
        ]
        post = self.s.posture.copy()
        post[:7] = q[:7]
        q, _ = k.solve_ik(q, tasks, active=self.s.active, posture=post, posture_w=self.s.posture_w, iters=iters)
        k.forward(q)
        fe = max(np.linalg.norm(k.d.xpos[self.fl] - tasks[0]["target"]),
                 np.linalg.norm(k.d.xpos[self.fr] - tasks[2]["target"]))
        ce = np.linalg.norm(k.com(q)[:2] - com)
        return q, fe, ce

    def _attempt(self, kind, seconds):
        k = self.k
        F, zmp, cmd, path, p = self._plan(kind, seconds)
        n = len(zmp)
        upper = self._upper_track(kind, n)
        pitch = 0.02 + 0.08 * np.clip(cmd[:, 0], 0, None)       # lean into forward walking a little
        # standing start: solve the first pose, then get the CoM height for the pendulum
        q = k.default_qpos()
        q[2] = p["h"]
        mid0 = 0.5 * (self._centre(F["L"][0]) + self._centre(F["R"][0]))
        for _ in range(6):
            q, fe, ce = self._solve(q, F["L"][0], F["R"][0], mid0, p["h"], upper[0], pitch[0], iters=30)
        if fe > 0.005 or ce > 0.01:
            return None, "start pose"
        zc = float(k.com(q)[2])
        com, _ = com_from_zmp(zmp, self.dt, zc, c0=mid0)
        frames = []
        for i in range(n):
            q, fe, ce = self._solve(q, F["L"][i], F["R"][i], com[i], p["h"], upper[i], pitch[i], iters=6)
            if fe > 0.01 or ce > 0.02:
                return None, f"IK ({'feet' if fe > 0.01 else 'centre of mass'}) at {i / self.fps:.1f} s"
            if self.check_every and (i % self.check_every == 0) and k.self_collisions(q):
                return None, f"self-collision at {i / self.fps:.1f} s"
            frames.append(q.copy())
        Q = np.array(frames)
        jv = np.max(np.abs(np.diff(Q[:, k.q_adr], axis=0))) * self.fps
        if jv > self.max_joint_vel:
            return None, f"joint speed {jv:.1f} rad/s"
        info = {"cmd": cmd, "path": path, "zmp": zmp, "com": com, "params": p, "zc": zc, "max_joint_vel": jv}
        return (Q[:, :7].copy(), Q[:, k.q_adr].copy(), info), "ok"

    def clip(self, kind="walk", seconds=20.0, max_tries=30, return_info=False):
        if kind not in LOCO_KINDS:
            raise ValueError(f"kind must be one of {LOCO_KINDS}")
        why = []
        for _ in range(max_tries):
            self.stats["attempts"] += 1
            out, msg = self._attempt(kind, seconds)
            if out is not None:
                self.stats["clips"] += 1
                root, joints, info = out
                return (root, joints, info) if return_info else (root, joints)
            why.append(msg)
        raise RuntimeError(f"could not make a '{kind}' clip in {max_tries} tries: {why[-5:]}")

    def library(self, kinds=LOCO_KINDS, clips_per_kind=20, seconds=20.0):
        clips, names, kk = [], [], []
        for kind in kinds:
            for i in range(clips_per_kind):
                clips.append(self.clip(kind, seconds))
                names.append(f"{kind}_{i:03d}")
                kk.append(kind)
                if self.verbose:
                    print(f"  {kind:8s} clip {i + 1}/{clips_per_kind}: {len(clips[-1][0]) / self.fps:.1f} s", flush=True)
        return clips, names, kk
