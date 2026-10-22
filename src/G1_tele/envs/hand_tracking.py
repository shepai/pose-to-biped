"""
src/G1_tele/envs/hand_tracking.py

Hand-target tracking on top of
  * the 29-DoF G1Env (my earlier g1_env.py - save it as G1_tele/envs/whole_body.py), and
  * your frozen pretrained leg policy (G1_tele.policies).

RL action : [waist(3) + arms(14)] joint-target offsets, plus optional 12 leg residuals.
Legs      : frozen policy output (+ leg_residual * residual action).

Dataset format (.npy / .npz): float array (T, 2, 3) per clip  ->  [left, right] hand position
RELATIVE TO ITS OWN SHOULDER, in the body frame (x forward, y left, z up), metres.
Several clips: .npz with one array per clip, or an object array of (T,2,3) arrays.
Positions are scaled by robot_arm_len / human_arm_len and clipped to max_reach.
"""
from pathlib import Path

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

import G1_tele.policies as policies
from G1_tele.envs.whole_body import G1Env
from G1_tele.policies import TorchScriptPolicy, build_policy_obs

DEFAULT_POLICY = Path(policies.__file__).resolve().parent / "motion.pt"

# Layout of the 29 controlled joints (defined here so this file is self-contained)
LEG_SLICE = slice(0, 12)       # L hip_pitch/roll/yaw, knee, ankle_pitch/roll, then R
WAIST_SLICE = slice(12, 15)    # yaw, roll, pitch
ARM_SLICE = slice(15, 29)      # L shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw, then R
N_JOINTS = 29
N_UP = 17                      # waist 3 + arms 14 = what the RL policy controls


class HandTargets:
    def __init__(self, path, fps, human_arm_len=0.65, robot_arm_len=0.50, max_reach=0.52):
        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.lib.npyio.NpzFile):
            clips = [raw[k] for k in raw.files]
        elif raw.dtype == object:
            clips = list(raw)
        elif raw.ndim == 3:
            clips = [raw]
        elif raw.ndim == 2 and raw.shape[1] == 6:
            clips = [raw.reshape(-1, 2, 3)]
        else:
            raise ValueError(f"Unsupported dataset shape {raw.shape}; expected (T,2,3)")
        s = robot_arm_len / human_arm_len
        self.clips = []
        for c in clips:
            c = np.asarray(c, np.float32).reshape(-1, 2, 3) * s
            n = np.linalg.norm(c, axis=-1, keepdims=True)
            self.clips.append(c * np.minimum(1.0, max_reach / np.maximum(n, 1e-6)))
        self.fps = fps

    def duration(self, i):
        return (len(self.clips[i]) - 1) / self.fps

    def get(self, i, t):
        c = self.clips[i]
        x = np.clip(t * self.fps, 0, len(c) - 1)
        k = int(np.floor(x)); k2 = min(k + 1, len(c) - 1); f = x - k
        return (1 - f) * c[k] + f * c[k2]                # (2, 3)


class HandTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, dataset_path, data_fps=30, xml_path=None,
                 leg_policy_path=DEFAULT_POLICY, leg_residual=0.0, push_force=0.0,
                 hand_bodies=("left_wrist_yaw_link", "right_wrist_yaw_link"),
                 hand_offset=(0.08, 0.0, 0.0), pelvis_body="pelvis",
                 shoulder_joints=("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
                 episode_seconds=10.0, lookahead=(0.1, 0.3, 0.5),
                 render_mode=None, seed=None):
        super().__init__()
        kw = dict(render_mode=render_mode, max_episode_seconds=1e9)
        if xml_path:
            kw["xml_path"] = xml_path
        self.base = G1Env(**kw)
        if self.base.n_joints != N_JOINTS:
            raise ValueError(f"Hand tracking needs the {N_JOINTS}-joint model (legs+waist+arms), "
                             f"but the loaded xml exposes {self.base.n_joints}. Pass a 29-DoF xml.")
        self.base.action_scale[LEG_SLICE] = 0.25         # matches the pretrained policy
        self.base.action_scale[WAIST_SLICE] = 0.3
        self.base.action_scale[ARM_SLICE] = 0.8          # arms need a wider range

        self.leg_policy = TorchScriptPolicy(str(leg_policy_path))
        self.hand_data = HandTargets(dataset_path, data_fps)
        self.rng = np.random.default_rng(seed)
        self.leg_residual = leg_residual
        self.n_res = 12 if leg_residual > 0 else 0
        self.push_force = push_force
        self.episode_seconds = episode_seconds
        self.lookahead = lookahead
        self.hand_offset = np.array(hand_offset)

        m = self.base.m
        self.pelvis_id = m.body(pelvis_body).id
        self.hand_ids = [m.body(n).id for n in hand_bodies]
        self.shoulder_ids = [m.joint(n).id for n in shoulder_joints]

        n_obs = self.base.observation_space.shape[0] + 6 * (3 + len(lookahead))
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (N_UP + self.n_res,), np.float32)
        self.render_mode = render_mode
        self.reset(seed=seed)

    # ------------------------------------------------------------ geometry
    def _R(self):
        return self.base.d.xmat[self.pelvis_id].reshape(3, 3)

    def _to_pelvis(self, p_world):                         # (n,3) world -> pelvis frame
        return (p_world - self.base.d.xpos[self.pelvis_id]) @ self._R()

    def _hands_world(self):
        d = self.base.d
        return np.stack([d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.hand_offset for b in self.hand_ids])

    def _targets_pelvis(self, t):
        return self.anchor + self.hand_data.get(self.clip, t)   # (2,3)

    def _targets_world(self, t):
        return self.base.d.xpos[self.pelvis_id] + self._targets_pelvis(t) @ self._R().T

    # ------------------------------------------------------------ obs
    def _obs(self):
        t = self.t
        hands = self._to_pelvis(self._hands_world())
        now = self._targets_pelvis(t)
        future = [self._targets_pelvis(t + dt).ravel() for dt in self.lookahead]
        return np.concatenate([self.base._obs(), hands.ravel(), now.ravel(),
                               (now - hands).ravel(), *future]).astype(np.float32)

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.base.reset()
        self.leg_policy.reset()
        self.last_policy_a = np.zeros(12, np.float32)
        self.prev_action = np.zeros(self.action_space.shape, np.float32)
        d = self.base.d
        anchors_world = np.stack([d.xanchor[j] for j in self.shoulder_ids])
        self.anchor = self._to_pelvis(anchors_world)       # shoulder positions in pelvis frame
        self.clip = int(self.rng.integers(len(self.hand_data.clips)))
        dur = self.hand_data.duration(self.clip)
        self.t = float(self.rng.uniform(0, max(0.0, dur - self.episode_seconds)))
        self.t_end = self.t + min(self.episode_seconds, dur)
        self.h0 = float(d.qpos[2]); self.xy0 = d.qpos[:2].copy()
        self.push_left = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1, 1)
        b = self.base

        # frozen leg policy (command = stand still)
        a_base = self.leg_policy(build_policy_obs(b, np.zeros(3, np.float32), self.last_policy_a))
        a_leg = a_base.copy()
        if self.n_res:
            a_leg = a_leg + 4.0 * self.leg_residual * action[N_UP:]
        full = np.zeros(b.n_joints, np.float32)
        full[LEG_SLICE] = a_leg
        full[WAIST_SLICE.start:] = action[:N_UP]

        # random pushes for robustness
        if self.push_force > 0 and self.push_left == 0 and self.rng.random() < 1 / 150:
            ang = self.rng.uniform(0, 2 * np.pi)
            b.d.xfrc_applied[self.pelvis_id, :3] = (self.push_force * self.rng.uniform(0.3, 1)
                                                    * np.array([np.cos(ang), np.sin(ang), 0]))
            self.push_left = 5
        elif self.push_left > 0:
            self.push_left -= 1
            if self.push_left == 0:
                b.d.xfrc_applied[self.pelvis_id, :3] = 0

        _, _, fell, _, _ = b.step(full)
        self.last_policy_a = a_base.astype(np.float32)
        self.t += b.dt

        # reward
        err = np.linalg.norm(self._hands_world() - self._targets_world(self.t), axis=1)
        s = b.state()
        g = s["gravity"]
        rew = (np.exp(-8.0 * err).sum() - 0.5 * err.sum()
               - 2.0 * np.sum(g[:2] ** 2)
               - 5.0 * (b.d.qpos[2] - self.h0) ** 2
               - 0.5 * np.sum(s["lin_vel"][:2] ** 2)
               - 0.2 * np.linalg.norm(b.d.qpos[:2] - self.xy0)
               - 0.01 * np.sum((action - self.prev_action) ** 2)
               - 1e-5 * np.sum(b.d.ctrl[b.act_adr] ** 2)
               + 0.5)
        if self.n_res:
            rew -= 0.05 * np.sum(action[N_UP:] ** 2)
        if fell:
            rew -= 10.0
        self.prev_action = action
        truncated = self.t >= self.t_end
        return self._obs(), float(rew), bool(fell), bool(truncated), {"hand_err": err}

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
    def frame_skip(self):                    # sim steps per policy step (dt = frame_skip * model.opt.timestep)
        return self.base.decim

    def draw_targets(self, scene):
        """Add the current left (red) / right (blue) target spheres to any mjvScene
        (viewer.user_scn or mujoco.Renderer.scene)."""
        tw = self._targets_world(self.t)
        for i, rgba in enumerate(([1.0, 0.2, 0.2, 0.7], [0.2, 0.4, 1.0, 0.7])):
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.03, 0, 0]), tw[i], np.eye(3).ravel(),
                                np.array(rgba, np.float32))
            scene.ngeom += 1

    def render(self):
        self.base.render()
        v = self.base.viewer
        if v is not None:
            v.user_scn.ngeom = 0
            self.draw_targets(v.user_scn)

    def close(self):
        self.base.close()