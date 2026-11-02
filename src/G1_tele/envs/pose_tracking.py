"""
src/G1_tele/envs/pose_tracking.py

Whole-body pose tracking for the 29-DoF G1: follow keypoint targets (hands, feet, and optionally
elbows, knees, hips, shoulders) while keeping balance. One policy controls every joint - there is
no frozen walking policy underneath, because a stepping gait cannot hold a foot where you ask it.

Balance comes first, pose accuracy second
-----------------------------------------
  * Reward = alive bonus + balance terms + tracking terms - small penalties (REWARD_WEIGHTS;
    override any of them with reward_weights={...}). With the defaults, staying up and balanced
    is worth 60 % of the per-step reward, matching the pose 40 %.
      balance:  capture point inside the support polygon of the feet on the ground (the term that
                warns about a fall ~0.5 s before it happens), pelvis tilt, pelvis height, low
                roll/pitch rate, feet on the ground when the reference has them planted
      tracking: keypoints (sigma_kp tolerance), joint angles, joint velocities
  * Privileged critic: the observation ends with balance state the real robot cannot measure
    (base velocity, centre of mass, capture point margin, foot contacts, friction, pushes). The
    PoseTrackingPolicy actor only sees the first n_actor_obs entries; the critic sees all of it,
    so it can predict falls and give the actor much cleaner learning signals.
  * Curriculum hooks: set_kind_weights() changes which clip kinds are sampled, and every episode
    reports info["clip_kind"] and info["survival"] (fraction of the possible episode survived).
    examples/run_logging.CurriculumCallback uses them to unlock harder motions as balance improves.

Other parts of the recipe (DeepMimic / H2O, scaled down for SB3):
  * Reference motions come from a motion library (G1_tele.motion.library): full robot
    configurations, by default procedurally generated balanced poses (examples/make_motions.py),
    or keypoint clips you retarget with G1_tele.motion.retarget.
  * Reference state initialisation: each episode starts at a random time of a random clip with the
    robot placed exactly in the reference pose.
  * Early termination: when the robot falls or the mean keypoint error exceeds max_kp_err.
  * Gravity feedforward (gravity_ff=True, residual mode): the PD target also gets tau_static / KP,
    the torques that hold the reference pose against gravity with the weight on the stance feet.
  * Balance assist (optional, off by default): stabilising wrench on the pelvis, annealed to zero.

Action (n_ctrl values in [-1, 1], n_ctrl = 23 with wrists locked, 29 otherwise):
    residual=True   PD target = reference joints (+ gravity offset) + action_scale * action  (default)
    residual=False  PD target = default pose + action_scale * action
Observation = [actor part (n_actor_obs) | privileged part (N_PRIVILEGED, only if privileged_obs)]
  actor part (all positions in the heading frame, see G1_tele.motion.kinematics):
    ang_vel*0.25, gravity, (q - default), dq*0.05, last action            (robot, proprioceptive)
    reference keypoints, reference - robot keypoints                      (targets for the next step)
    robot pelvis height, reference pelvis height, reference gravity vector
    [residual] reference joints - default, reference joints - q, reference joint velocity*0.05
    [lookahead] reference keypoints (and joints) at t + each lookahead time
Deploying: feed the actor obs[:n_actor_obs] (the privileged tail can be zeros - the actor ignores
it). residual=True needs reference joint angles at run time: run
G1_tele.motion.retarget.KeypointRetargeter on the tracker keypoints (~2 ms per frame).
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from G1_tele.envs.whole_body import DEFAULT_Q, WRIST_IDX, XML_PATH, G1Env
from G1_tele.motion.kinematics import (ANKLE_HEIGHT, FEET, KEYPOINTS, KEYPOINT_SETS, G1Kinematics, quat_to_mat, rot_z,
                                       rotvec_from_mat, yaw_of_quat)
from G1_tele.motion.library import DEFAULT_MOTIONS, MotionLibrary

#     residual action range (rad)  hip_p hip_r hip_y knee ank_p ank_r
_LEG_RES = np.array([0.5, 0.4, 0.3, 0.5, 0.7, 0.3] * 2)      # ankles/hips: enough torque to move the
_LEG_ABS = np.array([1.0, 0.6, 0.6, 1.2, 0.7, 0.3] * 2)      # centre of pressure across the whole sole
RESIDUAL_SCALE = np.concatenate([_LEG_RES, [0.3, 0.2, 0.2], [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3] * 2])
ABSOLUTE_SCALE = np.concatenate([_LEG_ABS, [0.6, 0.4, 0.4], [1.6, 1.4, 1.2, 1.2, 0.5, 0.5, 0.5] * 2])

KEYPOINT_WEIGHT = {"hand": 1.0, "foot": 1.0, "toe": 0.5, "elbow": 0.5, "knee": 0.5,
                   "shoulder": 0.5, "hip": 0.5, "head": 0.5}

# Positive terms sum to 1.0: alive 0.20 + balance 0.40 + tracking 0.40.
REWARD_WEIGHTS = dict(
    alive=0.20,
    cp=0.15,          # capture point inside the support polygon (with a 2 cm margin)
    upright=0.10,     # pelvis tilt matches the reference
    height=0.05,      # pelvis height matches the reference
    ang_vel=0.05,     # little pelvis roll/pitch rate (no wobbling)
    contact=0.05,     # feet on the ground when the reference has them planted (and lifted when lifted)
    kp=0.20,          # keypoint positions
    joint=0.12,       # joint angles
    joint_vel=0.08,   # joint velocities
    action_rate=0.08, torque=1e-6, slip=0.05,                 # penalties
)
SIGMAS = dict(cp=0.04, cp_margin=0.02, tilt=0.15, height=0.05, ang_vel=1.0, joint=0.3, joint_vel=1.5)

N_PRIVILEGED = 19
CONTACT_Z = 0.010            # a sole corner sphere (r = 5 mm) closer than this to the floor is in contact
LEFT_RGBA, RIGHT_RGBA, MID_RGBA = [1.0, 0.25, 0.2, 0.75], [0.2, 0.45, 1.0, 0.75], [0.2, 0.9, 0.3, 0.75]


def _names(spec):
    return list(KEYPOINT_SETS[spec]) if isinstance(spec, str) else list(spec)


# ---------------------------------------------------------------- 2-D support polygon helpers
def convex_hull(pts):
    """Monotone chain; pts (n,2) -> hull vertices counter-clockwise (n <= 8 here)."""
    p = sorted(map(tuple, pts))
    if len(p) <= 2:
        return np.array(p)

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    lower, upper = half(p), half(reversed(p))
    return np.array(lower[:-1] + upper[:-1])


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def support_margin(p, hull):
    """Signed distance from point p to the polygon: > 0 inside (how far from the edge), < 0 outside."""
    n = len(hull)
    if n == 0:
        return -1.0
    if n == 1:
        return -float(np.linalg.norm(p - hull[0]))
    if n == 2:
        return -_seg_dist(p, hull[0], hull[1])
    inside, dmin = True, np.inf
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0:
            inside = False
        dmin = min(dmin, _seg_dist(p, a, b))
    return dmin if inside else -dmin


class PoseTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, motion_file=None, xml_path=None, keypoints="full", obs_keypoints=None,
                 residual=True, control_wrists=False, episode_seconds=10.0, lookahead=(),
                 kind_weights=None, max_kp_err=0.30, push_force=0.0, push_interval=(2.0, 6.0),
                 friction_range=None, init_noise=0.03, obs_noise=0.0, action_scale=None,
                 sigma_kp=0.12, reward_weights=None, privileged_obs=True, gravity_ff=True,
                 assist=0.0, assist_anneal_steps=None, render_mode=None, seed=None):
        super().__init__()
        xml_path = xml_path or XML_PATH
        self.base = G1Env(xml_path, render_mode=render_mode, max_episode_seconds=1e9)
        self.kin = G1Kinematics(xml_path)                   # reference poses live here (never stepped)
        self.lib = MotionLibrary(motion_file or DEFAULT_MOTIONS)
        self.clip_kinds = sorted(set(str(k) for k in self.lib.kinds))

        self.kp_names = _names(keypoints)
        self.obs_kp_names = _names(obs_keypoints) if obs_keypoints is not None else self.kp_names
        self.kp_w = np.array([next(w for k, w in KEYPOINT_WEIGHT.items() if k in n) for n in self.kp_names])
        all_names = self.kp_names + [n for n in self.obs_kp_names if n not in self.kp_names]
        self._all_kp = all_names
        self._obs_sel = [all_names.index(n) for n in self.obs_kp_names]
        self.foot_kp = [all_names.index(n) for n in ("left_foot", "right_foot") if n in all_names]

        self.residual = residual
        self.gravity_ff = gravity_ff and residual
        self.ctrl = np.array([i for i in range(29) if control_wrists or i not in WRIST_IDX])
        self.n_ctrl = len(self.ctrl)
        scale = RESIDUAL_SCALE if residual else ABSOLUTE_SCALE
        self.action_scale = (np.asarray(action_scale, float) * np.ones(29) if action_scale is not None
                             else scale.copy())[self.ctrl]
        self.episode_seconds, self.lookahead = episode_seconds, tuple(lookahead)
        self.kind_weights = dict(kind_weights) if kind_weights else None
        self.max_kp_err, self.sigma_kp = max_kp_err, sigma_kp
        unknown = set(reward_weights or {}) - set(REWARD_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown reward_weights keys {unknown}; valid: {list(REWARD_WEIGHTS)}")
        self.w = {**REWARD_WEIGHTS, **(reward_weights or {})}
        self.push_force, self.push_interval = push_force, push_interval
        self.friction_range, self.init_noise, self.obs_noise = friction_range, init_noise, obs_noise
        self.privileged_obs = privileged_obs
        self.assist0, self.assist_anneal_steps = float(assist), assist_anneal_steps
        self.assist = self.assist0
        self.total_steps = 0                                # env steps taken over this env's lifetime
        self.push_wrench = np.zeros(6)
        self.friction_scale = 1.0

        m = self.base.m
        self.pelvis_id = m.body("pelvis").id
        self._kp_ids = np.array([m.body(KEYPOINTS[n][0]).id for n in all_names])
        self._kp_off = np.array([KEYPOINTS[n][1] for n in all_names], float)
        self._friction0 = m.geom_friction[:, 0].copy()
        self.q_lo, self.q_hi = self.base.q_lo + 0.02, self.base.q_hi - 0.02
        self._corners = []                                  # sole contact spheres, per foot
        for name in FEET:
            bid = m.body(name).id
            self._corners.append([g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g]])
        if not all(self._corners):
            raise ValueError("could not find collision geoms on the ankle_roll links (feet)")

        K, n = len(self.obs_kp_names), self.n_ctrl
        self.n_actor_obs = (6 + 3 * n + 6 * K + 5 + (3 * n if residual else 0)
                            + len(self.lookahead) * (3 * K + (n if residual else 0)))
        n_obs = self.n_actor_obs + (N_PRIVILEGED if privileged_obs else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.render_mode = render_mode
        self.reset(seed=seed)

    # ------------------------------------------------------------ curriculum hooks
    def set_kind_weights(self, weights):
        """Sampling weight per clip kind for the next episodes, e.g. {"upper": 1, "whole": 1}."""
        self.kind_weights = dict(weights) if weights else None

    def set_assist(self, value):
        """Fix the balance-assist strength (0 = off) and stop the automatic annealing."""
        self.assist0, self.assist, self.assist_anneal_steps = float(value), float(value), None

    # ------------------------------------------------------------ reference
    def _ref(self, t):
        """Reference state at time t: root, joints, jvel, local keypoints, height, gravity, foot contacts."""
        lib, c = self.lib, self.clip
        root, joints = lib.frame(c, t)
        _, _, jvel = lib.velocity(c, t)
        k = self.kin
        k.forward(k.qpos_from(root, joints))
        o, R = k.heading(root[:3], root[3:7])
        kps = (k.keypoints_world(self._all_kp) - o) @ R
        contact = k.d.xpos[k.foot_ids, 2] < ANKLE_HEIGHT + 0.015
        g = quat_to_mat(root[3:7]).T @ np.array([0.0, 0.0, -1.0])
        return {"root": root, "joints": joints, "jvel": jvel, "kp": kps, "h": root[2], "g": g, "contact": contact}

    def _gravity_offset(self, ref):
        """PD-target offset (tau_static / KP) that cancels gravity sag in the reference pose."""
        k = self.kin
        contacts = tuple(ref["contact"]) if ref["contact"].any() else (True, True)
        tau = k.static_torques(k.qpos_from(ref["root"], ref["joints"]), contacts)
        tau = np.clip(tau, -self.base.tau_lim, self.base.tau_lim)
        return np.clip(tau / self.base.kp, -0.5, 0.5)

    # ------------------------------------------------------------ robot
    def _robot_kp(self):
        d = self.base.d
        R = d.xmat[self._kp_ids].reshape(-1, 3, 3)
        p = d.xpos[self._kp_ids] + np.einsum("kij,kj->ki", R, self._kp_off)
        o = np.array([d.qpos[0], d.qpos[1], 0.0])
        return (p - o) @ rot_z(yaw_of_quat(d.qpos[3:7]))

    def _heading_to_world(self, p_local):
        d = self.base.d
        return np.array([d.qpos[0], d.qpos[1], 0.0]) + p_local @ rot_z(yaw_of_quat(d.qpos[3:7])).T

    def _balance_state(self):
        """CoM, CoM velocity, capture point margin and foot contacts. Needs kinematics + comPos done."""
        d = self.base.d
        com = d.subtree_com[self.pelvis_id].copy()
        com_vel = (com - self.prev_com) / self.base.dt
        self.prev_com = com
        pts, contact = [], []
        for corners in self._corners:
            z = d.geom_xpos[corners, 2]
            on = z < CONTACT_Z
            contact.append(bool(on.any()))
            pts.append(d.geom_xpos[corners][on, :2])
        pts = np.concatenate(pts)
        omega0 = np.sqrt(9.81 / max(com[2], 0.3))
        cp = com[:2] + com_vel[:2] / omega0
        margin = support_margin(cp, convex_hull(pts)) if len(pts) else -0.3
        self.bal = {"com": com, "com_vel": com_vel, "cp": cp, "margin": margin, "contact": np.array(contact)}
        return self.bal

    # ------------------------------------------------------------ obs
    def _obs(self):
        b, s = self.base, self.base.state()
        c, rn = self.ctrl, self.ref_next
        kp_now = self.robot_kp[self._obs_sel]
        kp_ref = rn["kp"][self._obs_sel]
        parts = [s["ang_vel"] * 0.25, s["gravity"], (s["q"] - b.default_q)[c], s["dq"][c] * 0.05, self.last_action,
                 kp_ref.ravel(), (kp_ref - kp_now).ravel(),
                 [b.d.qpos[2], rn["h"]], rn["g"]]
        if self.residual:
            parts += [(rn["joints"] - DEFAULT_Q)[c], (rn["joints"] - s["q"])[c], rn["jvel"][c] * 0.05]
        for dt in self.lookahead:
            r = self._ref(self.t + b.dt + dt)
            parts.append(r["kp"][self._obs_sel].ravel())
            if self.residual:
                parts.append((r["joints"] - DEFAULT_Q)[c])
        obs = np.concatenate(parts).astype(np.float32)
        if self.obs_noise > 0:
            obs += self.np_random.normal(0, self.obs_noise, obs.shape).astype(np.float32)
        if self.privileged_obs:
            d, bal = b.d, self.bal
            Rh = rot_z(yaw_of_quat(d.qpos[3:7]))
            o = np.array([d.qpos[0], d.qpos[1], 0.0])
            priv = np.concatenate([d.qvel[:3] @ Rh, (bal["com"] - o) @ Rh, bal["com_vel"] @ Rh,
                                   [np.clip(bal["margin"], -0.3, 0.3) * 10], bal["contact"], rn["contact"],
                                   [self.friction_scale], self.push_wrench[:3] / 100.0, [self.assist]])
            obs = np.concatenate([obs, priv.astype(np.float32)])
        return obs

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        b, rng = self.base, self.np_random
        b.reset()
        if options and "clip" in options:
            self.clip = int(options["clip"])
        else:
            self.clip = self.lib.sample_clip(rng, self.kind_weights, min_seconds=1.5)
        dur = self.lib.duration(self.clip)
        t0 = float(options["t0"]) if options and "t0" in options else float(rng.uniform(0, max(0.0, dur - 1.0)))
        self.t, self.t_end = t0, min(t0 + self.episode_seconds, dur)
        self.max_ep_steps = max(1, int(round((self.t_end - t0) / b.dt)))

        # reference state initialisation
        ref = self._ref(t0)
        lin, ang, jvel = self.lib.velocity(self.clip, t0)
        d = b.d
        d.qpos[:7] = ref["root"]
        d.qpos[2] += 0.002
        d.qpos[b.q_adr] = np.clip(ref["joints"] + rng.normal(0, self.init_noise, 29), b.q_lo, b.q_hi)
        d.qvel[:3], d.qvel[3:6] = lin, ang
        d.qvel[b.v_adr] = jvel
        mujoco.mj_forward(b.m, d)

        if self.friction_range is not None:
            self.friction_scale = float(rng.uniform(*self.friction_range))
            b.m.geom_friction[:, 0] = self._friction0 * self.friction_scale
        self.next_push = rng.uniform(*self.push_interval) if self.push_force > 0 else np.inf
        self.push_left = 0
        self.push_wrench[:] = 0.0
        b.d.xfrc_applied[:] = 0.0
        self.last_action = np.zeros(self.n_ctrl, np.float32)
        self.robot_kp = self._robot_kp()
        self.prev_foot = self.robot_kp[self.foot_kp].copy()
        self.prev_com = d.subtree_com[self.pelvis_id] - np.asarray(d.qvel[:3]) * b.dt  # com_vel ~ base vel
        self._balance_state()
        self.ref_next = self._ref(self.t + b.dt)
        self.ep_stats = {"kp_err": 0.0, "steps": 0}
        return self._obs(), {}

    def _push(self):
        if self.push_left > 0:
            self.push_left -= 1
            if self.push_left == 0:
                self.push_wrench[:3] = 0.0
        elif self.t >= self.next_push:
            ang = self.np_random.uniform(0, 2 * np.pi)
            f = self.push_force * self.np_random.uniform(0.3, 1.0)
            self.push_wrench[:3] = f * np.array([np.cos(ang), np.sin(ang), 0.0])
            self.push_left = int(self.np_random.integers(5, 11))
            self.next_push = self.t + self.np_random.uniform(*self.push_interval)

    def _assist_wrench(self, ref):
        """'Hand of god' on the pelvis: PD towards the reference pelvis height and roll/pitch.
        Yaw and horizontal position are never assisted."""
        d, w = self.base.d, np.zeros(6)
        if self.assist <= 0:
            return w
        R = quat_to_mat(d.qpos[3:7])
        yaw = yaw_of_quat(d.qpos[3:7])
        R_t = rot_z(yaw) @ rot_z(yaw_of_quat(ref["root"][3:7])).T @ quat_to_mat(ref["root"][3:7])
        e = rotvec_from_mat(R_t @ R.T)
        e[2] = 0.0
        omega = R @ d.qvel[3:6]
        w[3:5] = 250.0 * e[:2] - 15.0 * omega[:2]
        w[2] = 1500.0 * (ref["h"] - d.qpos[2]) - 100.0 * d.qvel[2]
        return self.assist * w

    def step(self, action):
        b = self.base
        self.total_steps += 1
        if self.assist_anneal_steps:
            self.assist = self.assist0 * max(0.0, 1.0 - self.total_steps / self.assist_anneal_steps)
        a = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
        ref = self.ref_next                                 # reference at the end of this control step
        target = (ref["joints"] if self.residual else DEFAULT_Q).astype(np.float64).copy()
        if self.gravity_ff:
            target += self._gravity_offset(ref)
        target[self.ctrl] += self.action_scale * a
        target = np.clip(target, self.q_lo, self.q_hi)

        self._push()
        if self.assist > 0:
            pid = self.pelvis_id

            def apply(d):                                   # 500 Hz so the stiff assist PD stays stable
                d.xfrc_applied[pid] = self.push_wrench + self._assist_wrench(ref)
            b.step_targets(target, apply)
        else:
            b.d.xfrc_applied[self.pelvis_id] = self.push_wrench
            b.step_targets(target)
        mujoco.mj_kinematics(b.m, b.d)
        mujoco.mj_comPos(b.m, b.d)
        self.t += b.dt
        self.robot_kp = kp = self._robot_kp()
        s = b.state()
        bal = self._balance_state()
        w, sg = self.w, SIGMAS

        # ---- balance terms (robot now vs reference at the same instant)
        r_cp = np.exp(-(max(0.0, sg["cp_margin"] - bal["margin"]) / sg["cp"]) ** 2)
        dg = np.linalg.norm(s["gravity"] - ref["g"])
        r_up = np.exp(-(dg / sg["tilt"]) ** 2)
        dh = b.d.qpos[2] - ref["h"]
        r_h = np.exp(-(dh / sg["height"]) ** 2)
        r_av = np.exp(-np.sum(s["ang_vel"][:2] ** 2) / sg["ang_vel"] ** 2)
        contact_match = float(np.mean(bal["contact"] == ref["contact"]))

        # ---- tracking terms
        c = self.ctrl
        nk = len(self.kp_names)
        e_kp = np.linalg.norm(kp[:nk] - ref["kp"][:nk], axis=1)
        kp_err = float(np.sum(self.kp_w * e_kp) / np.sum(self.kp_w))
        r_kp = np.exp(-np.sum(self.kp_w * e_kp ** 2) / np.sum(self.kp_w) / self.sigma_kp ** 2)
        e_q = s["q"][c] - ref["joints"][c]
        r_q = np.exp(-np.mean(e_q ** 2) / sg["joint"] ** 2)
        r_vel = np.exp(-np.mean((s["dq"][c] - ref["jvel"][c]) ** 2) / sg["joint_vel"] ** 2)

        # ---- penalties: planted reference feet should not slide, smooth actions, low torque
        foot = kp[self.foot_kp]
        slip = 0.0
        for i, fi in enumerate(self.foot_kp):
            if ref["kp"][fi, 2] < 0.05:
                slip += np.sum(((foot[i, :2] - self.prev_foot[i, :2]) / b.dt) ** 2)
        self.prev_foot = foot.copy()
        p_rate = np.mean((a - self.last_action) ** 2)
        p_tau = np.sum(b.d.ctrl[b.act_adr] ** 2)

        r_balance = w["cp"] * r_cp + w["upright"] * r_up + w["height"] * r_h + w["ang_vel"] * r_av \
            + w["contact"] * contact_match
        r_track = w["kp"] * r_kp + w["joint"] * r_q + w["joint_vel"] * r_vel
        rew = w["alive"] + r_balance + r_track - w["action_rate"] * p_rate - w["torque"] * p_tau - w["slip"] * slip
        self.last_action = a

        # fell: pelvis tilted > ~45 deg, or 25 cm lower than the reference; lost: pose too far off
        fell = bool(s["gravity"][2] > -0.7 or dh < -0.25)
        lost = (not fell) and kp_err > self.max_kp_err
        terminated = fell or lost
        truncated = (not terminated) and self.t >= self.t_end - 1e-6

        self.ep_stats["kp_err"] += kp_err
        self.ep_stats["steps"] += 1
        names = self.kp_names
        info = {"kp_err": kp_err, "joint_err": float(np.sqrt(np.mean(e_q ** 2))), "height_err": float(abs(dh)),
                "hand_err": float(np.mean([e_kp[i] for i, n in enumerate(names) if "hand" in n] or [np.nan])),
                "foot_err": float(np.mean([e_kp[i] for i, n in enumerate(names) if "foot" in n] or [np.nan])),
                "cp_margin": float(bal["margin"]), "contact_match": contact_match,
                "r_balance": float(r_balance), "r_track": float(r_track),
                "r_cp": float(r_cp), "r_upright": float(r_up), "r_kp": float(r_kp), "r_joint": float(r_q),
                "fell": fell, "lost": lost, "assist": self.assist}
        if terminated or truncated:
            info["episode_kp_err"] = self.ep_stats["kp_err"] / max(1, self.ep_stats["steps"])
            info["clip_kind"] = str(self.lib.kinds[self.clip])
            info["survival"] = 1.0 if not terminated else min(1.0, self.ep_stats["steps"] / self.max_ep_steps)
        self.ref_next = self._ref(self.t + b.dt)
        return self._obs(), float(rew), terminated, truncated, info

    # ------------------------------------------------------------ viewer / recording helpers
    @property
    def model(self):
        return self.base.m

    @property
    def data(self):
        return self.base.d

    @property
    def dt(self):
        return self.base.dt

    @property
    def frame_skip(self):
        return self.base.decim

    def current_targets_world(self):
        """Reference keypoints placed on the robot (the robot's heading frame), world coordinates."""
        r = self._ref(self.t)
        return self._heading_to_world(r["kp"][:len(self.kp_names)])

    def draw_targets(self, scene, radius=0.03):
        """Add target spheres (left = red, right = blue) to an mjvScene (viewer.user_scn or Renderer.scene)."""
        for name, p in zip(self.kp_names, self.current_targets_world()):
            if scene.ngeom >= scene.maxgeom:
                break
            rgba = LEFT_RGBA if name.startswith("left") else RIGHT_RGBA if name.startswith("right") else MID_RGBA
            r = radius if ("hand" in name or "foot" in name) else 0.6 * radius
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, 0, 0]), p, np.eye(3).ravel(), np.array(rgba, np.float32))
            scene.ngeom += 1

    def render(self):
        self.base.render()
        v = self.base.viewer
        if v is not None:
            with v.lock():
                v.user_scn.ngeom = 0
                self.draw_targets(v.user_scn)
            v.sync()

    def close(self):
        self.base.close()
