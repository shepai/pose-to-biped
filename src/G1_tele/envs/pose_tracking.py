"""
src/G1_tele/envs/pose_tracking.py

Whole-body pose tracking for the 29-DoF G1: follow keypoint targets (hands, feet, and optionally
elbows, knees, hips, shoulders) while keeping balance. One policy controls every joint - there is
no frozen walking policy underneath, because a stepping gait cannot hold a foot where you ask it.

How the task is set up (the standard recipe from DeepMimic / H2O / OmniH2O, scaled down for SB3):

  * Reference motions come from a motion library (G1_tele.motion.library): full robot
    configurations, by default procedurally generated balanced poses (examples/make_motions.py),
    or keypoint clips you retarget with G1_tele.motion.retarget.
  * Reference state initialisation: each episode starts at a random time of a random clip with the
    robot placed exactly in the reference pose, so every part of every clip gets practised.
  * Early termination: the episode ends when the robot falls or the mean keypoint error exceeds
    max_kp_err, so the policy never wastes samples flailing on the ground.
  * Balance assist (optional, assist=1.0 with assist_anneal_steps=N): a stabilising wrench on the
    pelvis towards the reference height and tilt, fading linearly to zero over N env steps. Early
    on the policy can learn what tracking means instead of falling within a second every episode.
  * Bounded reward (roughly 0..1 per step): a sum of exp(-error^2 / sigma^2) terms for keypoints,
    joint angles, pelvis height/tilt and joint velocities, minus small smoothness penalties.
    No large fall penalty, so the value function and the logged return are far less noisy.

  * Gravity feedforward (gravity_ff=True, residual mode): the PD target also gets tau_static / KP,
    the torques that hold the reference pose against gravity with the weight on the stance feet
    (G1Kinematics.static_torques). Without it the stiff-ish PD sags ~6 cm within 0.2 s and the
    policy first has to learn to undo that; with it the error at 0.2 s is ~1.6 cm.

Action (n_ctrl values in [-1, 1], n_ctrl = 23 with wrists locked, 29 otherwise):
    residual=True   PD target = reference joints (+ gravity offset) + action_scale * action  (default)
    residual=False  PD target = default pose + action_scale * action
Observation (all positions in the heading frame, see G1_tele.motion.kinematics):
    ang_vel*0.25, gravity, (q - default), dq*0.05, last action            (robot, proprioceptive)
    reference keypoints, reference - robot keypoints                      (targets for the next step)
    robot pelvis height, reference pelvis height, reference gravity vector
    [residual] reference joints - default, reference joints - q, reference joint velocity*0.05
    [lookahead] reference keypoints (and joints) at t + each lookahead time
Deploying residual=True needs reference joint angles at run time: run
G1_tele.motion.retarget.KeypointRetargeter on the tracker keypoints (~2 ms per frame).
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from G1_tele.envs.whole_body import (BODY_JOINTS, DEFAULT_Q, LEG_SLICE, WAIST_SLICE, ARM_SLICE, WRIST_IDX,
                                     XML_PATH, G1Env)
from G1_tele.motion.kinematics import (ANKLE_HEIGHT, KEYPOINTS, KEYPOINT_SETS, G1Kinematics, quat_to_mat, rot_z, rotvec_from_mat,
                                       yaw_of_quat)
from G1_tele.motion.library import DEFAULT_MOTIONS, MotionLibrary

#                         hip_p hip_r hip_y knee ank_p ank_r
_LEG_RES = np.array([0.4, 0.3, 0.3, 0.5, 0.4, 0.3] * 2)
_LEG_ABS = np.array([1.0, 0.6, 0.6, 1.2, 0.6, 0.3] * 2)
RESIDUAL_SCALE = np.concatenate([_LEG_RES, [0.3, 0.2, 0.2], [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3] * 2])
ABSOLUTE_SCALE = np.concatenate([_LEG_ABS, [0.6, 0.4, 0.4], [1.6, 1.4, 1.2, 1.2, 0.5, 0.5, 0.5] * 2])

KEYPOINT_WEIGHT = {"hand": 1.0, "foot": 1.0, "toe": 0.5, "elbow": 0.5, "knee": 0.5,
                   "shoulder": 0.5, "hip": 0.5, "head": 0.5}
LEFT_RGBA, RIGHT_RGBA, MID_RGBA = [1.0, 0.25, 0.2, 0.75], [0.2, 0.45, 1.0, 0.75], [0.2, 0.9, 0.3, 0.75]


def _names(spec):
    return list(KEYPOINT_SETS[spec]) if isinstance(spec, str) else list(spec)


class PoseTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, motion_file=None, xml_path=None, keypoints="full", obs_keypoints=None,
                 residual=True, control_wrists=False, episode_seconds=10.0, lookahead=(),
                 kind_weights=None, max_kp_err=0.25, push_force=0.0, push_interval=(2.0, 6.0),
                 friction_range=None, init_noise=0.03, obs_noise=0.0, action_scale=None,
                 sigma_kp=0.10, gravity_ff=True, assist=0.0, assist_anneal_steps=None, render_mode=None,
                 seed=None):
        super().__init__()
        xml_path = xml_path or XML_PATH
        self.base = G1Env(xml_path, render_mode=render_mode, max_episode_seconds=1e9)
        self.kin = G1Kinematics(xml_path)                   # reference poses live here (never stepped)
        self.lib = MotionLibrary(motion_file or DEFAULT_MOTIONS)

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
        self.episode_seconds, self.lookahead, self.kind_weights = episode_seconds, tuple(lookahead), kind_weights
        self.max_kp_err, self.sigma_kp = max_kp_err, sigma_kp
        self.push_force, self.push_interval = push_force, push_interval
        self.friction_range, self.init_noise, self.obs_noise = friction_range, init_noise, obs_noise
        self.assist0, self.assist_anneal_steps = float(assist), assist_anneal_steps
        self.assist = self.assist0
        self.total_steps = 0                                # env steps taken over this env's lifetime
        self.push_wrench = np.zeros(6)

        m = self.base.m
        self.pelvis_id = m.body("pelvis").id
        self._kp_ids = np.array([m.body(KEYPOINTS[n][0]).id for n in all_names])
        self._kp_off = np.array([KEYPOINTS[n][1] for n in all_names], float)
        self._friction0 = m.geom_friction[:, 0].copy()
        self.q_lo, self.q_hi = self.base.q_lo + 0.02, self.base.q_hi - 0.02

        K, n = len(self.obs_kp_names), self.n_ctrl
        n_obs = 6 + 3 * n + 6 * K + 5 + (3 * n if residual else 0) + len(self.lookahead) * (3 * K + (n if residual else 0))
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.render_mode = render_mode
        self.reset(seed=seed)

    # ------------------------------------------------------------ reference
    def _ref(self, t):
        """Reference state at time t: dict with root, joints, jvel, local keypoints, height, gravity."""
        lib, c = self.lib, self.clip
        root, joints = lib.frame(c, t)
        _, _, jvel = lib.velocity(c, t)
        k = self.kin
        k.forward(k.qpos_from(root, joints))
        o, R = k.heading(root[:3], root[3:7])
        kps = (k.keypoints_world(self._all_kp) - o) @ R
        feet_z = k.d.xpos[k.foot_ids, 2].copy()
        g = quat_to_mat(root[3:7]).T @ np.array([0.0, 0.0, -1.0])
        return {"root": root, "joints": joints, "jvel": jvel, "kp": kps, "h": root[2], "g": g, "feet_z": feet_z}

    def _gravity_offset(self, ref):
        """PD-target offset (tau_static / KP) that cancels gravity sag in the reference pose."""
        k = self.kin
        contacts = tuple(ref["feet_z"] < ANKLE_HEIGHT + 0.015)
        if not any(contacts):
            contacts = (True, True)
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
        return obs

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        b, rng = self.base, self.np_random
        b.reset()
        self.clip = self.lib.sample_clip(rng, self.kind_weights, min_seconds=1.5)
        dur = self.lib.duration(self.clip)
        if options and "clip" in options:
            self.clip = int(options["clip"]); dur = self.lib.duration(self.clip)
        t0 = float(options.get("t0", 0.0)) if options and "t0" in options else float(rng.uniform(0, max(0.0, dur - 1.0)))
        self.t, self.t_end = t0, min(t0 + self.episode_seconds, dur)

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
            b.m.geom_friction[:, 0] = self._friction0 * rng.uniform(*self.friction_range)
        self.next_push = rng.uniform(*self.push_interval) if self.push_force > 0 else np.inf
        self.push_left = 0
        self.push_wrench[:] = 0.0
        b.d.xfrc_applied[:] = 0.0
        self.last_action = np.zeros(self.n_ctrl, np.float32)
        self.robot_kp = self._robot_kp()
        self.prev_foot = self.robot_kp[self.foot_kp].copy()
        self.ref_next = self._ref(self.t + b.dt)
        self.ep_stats = {"kp_err": 0.0, "steps": 0}
        return self._obs(), {}

    def set_assist(self, value):
        """Fix the balance-assist strength (0 = off) and stop the automatic annealing."""
        self.assist0, self.assist, self.assist_anneal_steps = float(value), float(value), None

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
        Only used early in training (annealed to 0) so the policy can learn tracking before it has
        learned to balance; yaw and horizontal position are never assisted."""
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
        self.t += b.dt
        self.robot_kp = kp = self._robot_kp()
        s = b.state()

        # ---- reward: robot now vs reference at the same instant
        c = self.ctrl
        e_kp = np.linalg.norm(kp[:len(self.kp_names)] - ref["kp"][:len(self.kp_names)], axis=1)
        kp_err = float(np.sum(self.kp_w * e_kp) / np.sum(self.kp_w))
        r_kp = np.exp(-np.sum(self.kp_w * e_kp ** 2) / np.sum(self.kp_w) / self.sigma_kp ** 2)
        e_q = s["q"][c] - ref["joints"][c]
        r_q = np.exp(-np.mean(e_q ** 2) / 0.3 ** 2)
        dh = b.d.qpos[2] - ref["h"]
        dg = np.linalg.norm(s["gravity"] - ref["g"])
        r_root = np.exp(-(dh / 0.05) ** 2 - (dg / 0.15) ** 2)
        r_vel = np.exp(-np.mean((s["dq"][c] - ref["jvel"][c]) ** 2) / 1.5 ** 2)
        # planted reference feet should not slide
        foot = kp[self.foot_kp]
        slip = 0.0
        for i, fi in enumerate(self.foot_kp):
            if ref["kp"][fi, 2] < 0.05:
                slip += np.sum(((foot[i, :2] - self.prev_foot[i, :2]) / b.dt) ** 2)
        self.prev_foot = foot.copy()
        rew = (0.40 * r_kp + 0.25 * r_q + 0.20 * r_root + 0.10 * r_vel + 0.05
               - 0.08 * np.mean((a - self.last_action) ** 2)
               - 1e-6 * np.sum(b.d.ctrl[b.act_adr] ** 2)
               - 0.05 * slip)
        self.last_action = a

        fell = bool(b.d.qpos[2] < 0.3 or s["gravity"][2] > -0.5)
        lost = kp_err > self.max_kp_err
        terminated = fell or lost
        truncated = (not terminated) and self.t >= self.t_end - 1e-6

        self.ep_stats["kp_err"] += kp_err; self.ep_stats["steps"] += 1
        names = self.kp_names
        info = {"kp_err": kp_err, "joint_err": float(np.sqrt(np.mean(e_q ** 2))), "height_err": float(abs(dh)),
                "hand_err": float(np.mean([e_kp[i] for i, n in enumerate(names) if "hand" in n] or [np.nan])),
                "foot_err": float(np.mean([e_kp[i] for i, n in enumerate(names) if "foot" in n] or [np.nan])),
                "r_kp": float(r_kp), "r_joint": float(r_q), "r_root": float(r_root), "fell": fell, "lost": lost,
                "assist": self.assist}
        if terminated or truncated:
            info["episode_kp_err"] = self.ep_stats["kp_err"] / max(1, self.ep_stats["steps"])
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
