"""
Policy-agnostic Gymnasium environment for the Unitree G1 (12 leg motors) in MuJoCo.

The model is loaded from   <package>/assets/g1_description/scene.xml
No yaml, no policy loading here - that lives in whatever script drives the env.

Controlled joints (order used for `action`, `state()["q"]`, ...):
    0-5    left leg   (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    6-11   right leg  (same order)

action : 12 values, target joint position = default_q + action_scale * action
         (0.25 scale and PD gains match unitree_rl_gym, so a pretrained policy's raw output plugs straight in)
obs    : [ang_vel*0.25 (3), gravity (3), (q-default)(12), dq*0.05 (12), last_action (12)] = 42
reward : 0.0 (put your task reward in a wrapper); terminated when the robot falls over.
"""
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

XML_PATH = Path(__file__).resolve().parent.parent / "assets" / "g1_description" / "scene.xml"

LEG = [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
N_JOINTS = len(LEG)                      # 12
LEG_SLICE = slice(0, N_JOINTS)           # kept so existing scripts that import it still work

DEFAULT_Q = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, np.float32)
KP = np.array([100, 100, 100, 150, 40, 40] * 2, np.float32)
KD = np.array([2, 2, 2, 4, 2, 2] * 2, np.float32)
TAU_LIM = np.array([88, 139, 88, 139, 50, 50] * 2, np.float32)


class G1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, xml_path=XML_PATH, render_mode=None, key_callback=None,
                 sim_dt=0.002, decimation=10, action_scale=0.25,
                 max_episode_seconds=20.0, fall_height=0.5, fall_tilt=-0.6):
        super().__init__()
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.m.opt.timestep = sim_dt
        self.d = mujoco.MjData(self.m)
        self._to_torque_mode()

        self.sim_dt, self.decim = sim_dt, decimation
        self.dt = sim_dt * decimation                         # 0.02 s -> 50 Hz policy rate
        self.max_steps = int(max_episode_seconds / self.dt)
        self.fall_height, self.fall_tilt = fall_height, fall_tilt

        self.n_joints = N_JOINTS
        self.joint_names = list(LEG)
        self.default_q = DEFAULT_Q.copy()
        self.kp, self.kd, self.tau_lim = KP, KD, TAU_LIM
        self.action_scale = np.full(N_JOINTS, action_scale, np.float32)   # per-joint, editable

        j = lambda n: self.m.joint(n)
        self.q_adr = np.array([j(n).qposadr[0] for n in LEG])
        self.v_adr = np.array([j(n).dofadr[0] for n in LEG])
        act_for = {int(self.m.actuator_trnid[i, 0]): i for i in range(self.m.nu)}
        self.act_adr = np.array([act_for[j(n).id] for n in LEG])

        self.observation_space = spaces.Box(-np.inf, np.inf, (6 + 3 * N_JOINTS,), np.float32)
        self.action_space = spaces.Box(-10.0, 10.0, (N_JOINTS,), np.float32)

        self.render_mode, self._key_cb, self.viewer = render_mode, key_callback, None
        self.sim_steps = self.steps = 0
        self.last_action = np.zeros(N_JOINTS, np.float32)

    # ------------------------------------------------------------------ setup
    def _to_torque_mode(self):
        """Make all actuators plain torque motors; PD is computed here."""
        m = self.m
        m.actuator_gaintype[:] = 0
        m.actuator_biastype[:] = 0
        m.actuator_gainprm[:, :] = 0
        m.actuator_gainprm[:, 0] = 1
        m.actuator_biasprm[:, :] = 0
        m.actuator_ctrllimited[:] = 0
        m.actuator_forcelimited[:] = 0

    # ------------------------------------------------------------------ state
    def _rot(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.d.qpos[3:7])
        return R.reshape(3, 3)                                # body -> world

    def state(self):
        """Raw (unscaled) state, for building policy-specific observations."""
        d = self.d
        R = self._rot()
        return {
            "ang_vel": d.qvel[3:6].astype(np.float32),                     # body frame
            "gravity": (R.T @ np.array([0, 0, -1.0])).astype(np.float32),  # gravity in body frame
            "lin_vel": (R.T @ d.qvel[0:3]).astype(np.float32),             # body frame
            "q": d.qpos[self.q_adr].astype(np.float32),
            "dq": d.qvel[self.v_adr].astype(np.float32),
            "base_pos": d.qpos[0:3].astype(np.float32),
            "time": self.sim_steps * self.sim_dt,
        }

    def _obs(self):
        s = self.state()
        return np.concatenate([s["ang_vel"] * 0.25, s["gravity"], s["q"] - self.default_q,
                               s["dq"] * 0.05, self.last_action]).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)
        if self.d.qpos[2] < 0.5:                              # xml gave no standing height
            self.d.qpos[2] = 0.8
        self.d.qpos[self.q_adr] = self.default_q
        mujoco.mj_forward(self.m, self.d)
        self.sim_steps = self.steps = 0
        self.last_action = np.zeros(N_JOINTS, np.float32)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -10, 10)
        target = self.default_q + self.action_scale * action
        d = self.d
        for _ in range(self.decim):
            tau = self.kp * (target - d.qpos[self.q_adr]) - self.kd * d.qvel[self.v_adr]
            d.ctrl[self.act_adr] = np.clip(tau, -self.tau_lim, self.tau_lim)
            mujoco.mj_step(self.m, d)
            self.sim_steps += 1
        self.steps += 1
        self.last_action = action

        s = self.state()
        terminated = bool(s["base_pos"][2] < self.fall_height or s["gravity"][2] > self.fall_tilt)
        truncated = self.steps >= self.max_steps
        return self._obs(), 0.0, terminated, truncated, {"state": s}

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d, key_callback=self._key_cb)
        self.viewer.sync()

    def viewer_is_running(self):
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None