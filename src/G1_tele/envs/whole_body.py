"""
Policy-agnostic Gymnasium environment for the Unitree G1 (29 DoF, optional hands) in MuJoCo.

The model is loaded from   G1_tele/assets/g1_description/g1_29dof.xml   (override with xml_path=)
The xml must contain a floor (unitree_rl_gym's g1_29dof.xml does).

Joint order of the 29 controlled body joints (used for `action`, `state()["q"]`, ...):
    0-11   legs   (L hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll, then R)
    12-14  waist  (yaw, roll, pitch)
    15-21  left arm  (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
    22-28  right arm
Hand (finger) joints are not part of the action; they are PD-held at 0.

action : 29 values, target joint position = default_q + action_scale * action
         (or call step_targets(q_target) to give absolute PD targets directly)
obs    : [ang_vel*0.25 (3), gravity (3), (q-default)(29), dq*0.05 (29), last_action (29)] = 93
reward : 0.0 (put your task reward in a wrapper); terminated when the robot falls over.

Joint armature (motor rotor inertia)
------------------------------------
unitree_rl_gym's g1_29dof.xml ships with armature = 0 on every joint, while the 12-DoF model the
pretrained walking policy was trained on has armature = 0.01. With zero armature the PD loop on the
light links behaves very differently and that policy falls over within ~1.5 s, even with the arms
perfectly still. armature="auto" (default) fills in realistic per-motor values whenever the xml
has none; pass a float to force a uniform value, or None to leave the xml untouched.
"""
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

XML_PATH = Path(__file__).resolve().parent.parent / "assets" / "g1_description" / "g1_29dof.xml"

LEG = [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM = [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")]
BODY_JOINTS = LEG + WAIST + ARM          # 29
LEG_SLICE, WAIST_SLICE, ARM_SLICE = slice(0, 12), slice(12, 15), slice(15, 29)
WRIST_IDX = [19, 20, 21, 26, 27, 28]

_arm_default = lambda sgn: [0.0, 0.15 * sgn, 0.0, 0.3, 0.0, 0.0, 0.0]
DEFAULT_Q = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0, 0, 0]
                     + _arm_default(1) + _arm_default(-1), np.float32)

KP = np.array([100, 100, 100, 150, 40, 40] * 2 + [300, 300, 300]
              + [60, 60, 60, 60, 30, 30, 30] * 2, np.float32)
KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [5, 5, 5]
              + [2, 2, 2, 2, 1, 1, 1] * 2, np.float32)
TAU_LIM = np.array([88, 139, 88, 139, 50, 50] * 2 + [88, 50, 50]
                   + [25, 25, 25, 25, 25, 5, 5] * 2, np.float32)
HAND_KP, HAND_KD, HAND_TAU = 3.0, 0.2, 3.0

# Reflected rotor inertia of the G1 motors (kg m^2): 7520-14, 7520-22, 5020 (x2 for the
# parallel-linkage ankles / waist roll+pitch) and 4010 for the small wrist motors.
_A14, _A22, _A50, _A40 = 0.010177520, 0.025101925, 0.003609725, 0.00425
ARMATURE = np.array([_A14, _A22, _A14, _A22, 2 * _A50, 2 * _A50] * 2 + [_A14, 2 * _A50, 2 * _A50]
                    + [_A50, _A50, _A50, _A50, _A50, _A40, _A40] * 2, np.float32)


class G1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, xml_path=XML_PATH, render_mode=None, key_callback=None,
                 sim_dt=0.002, decimation=10, action_scale=0.25,
                 max_episode_seconds=20.0, fall_height=0.5, fall_tilt=-0.6, armature="auto"):
        super().__init__()
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.m.opt.timestep = sim_dt
        self.d = mujoco.MjData(self.m)
        self._to_torque_mode()

        self.sim_dt, self.decim = sim_dt, decimation
        self.dt = sim_dt * decimation
        self.action_scale = np.full(len(BODY_JOINTS), action_scale, np.float32)  # per-joint, editable
        self.max_steps = int(max_episode_seconds / self.dt)
        self.fall_height, self.fall_tilt = fall_height, fall_tilt

        j = lambda n: self.m.joint(n)
        self.n_joints = len(BODY_JOINTS)
        self.q_adr = np.array([j(n).qposadr[0] for n in BODY_JOINTS])
        self.v_adr = np.array([j(n).dofadr[0] for n in BODY_JOINTS])
        act_for = {int(self.m.actuator_trnid[i, 0]): i for i in range(self.m.nu)}
        self.act_adr = np.array([act_for[j(n).id] for n in BODY_JOINTS])
        self.q_lo = np.array([j(n).range[0] for n in BODY_JOINTS], np.float32)
        self.q_hi = np.array([j(n).range[1] for n in BODY_JOINTS], np.float32)

        # every other actuated joint (fingers) is held at 0
        body_ids = {j(n).id for n in BODY_JOINTS}
        hand = [(jid, a) for jid, a in act_for.items() if jid not in body_ids]
        self.hand_q = np.array([self.m.jnt_qposadr[jid] for jid, _ in hand], int)
        self.hand_v = np.array([self.m.jnt_dofadr[jid] for jid, _ in hand], int)
        self.hand_act = np.array([a for _, a in hand], int)

        if armature is not None:
            arm = self.m.dof_armature
            if isinstance(armature, str):                 # "auto": only fill in if the xml has none
                if np.all(arm[self.v_adr] == 0):
                    arm[self.v_adr] = ARMATURE
            else:
                arm[self.v_adr] = armature

        if not any(self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in range(self.m.ngeom)):
            raise ValueError(f"{xml_path} has no floor plane. Use a scene xml that includes the robot "
                             "and a <geom type='plane'/> floor (unitree_rl_gym's g1_29dof.xml has one).")

        self.default_q = DEFAULT_Q.copy()
        self.kp, self.kd, self.tau_lim = KP.copy(), KD.copy(), TAU_LIM.copy()
        self.observation_space = spaces.Box(-np.inf, np.inf, (6 + 3 * self.n_joints,), np.float32)
        self.action_space = spaces.Box(-10.0, 10.0, (self.n_joints,), np.float32)

        self.render_mode, self._key_cb, self.viewer = render_mode, key_callback, None
        self.sim_steps = self.steps = 0
        self.last_action = np.zeros(self.n_joints, np.float32)

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
        return R.reshape(3, 3)                       # body -> world

    def state(self):
        """Raw (unscaled) state, for building policy-specific observations."""
        d = self.d
        R = self._rot()
        return {
            "ang_vel": d.qvel[3:6].astype(np.float32),                    # body frame
            "gravity": (R.T @ np.array([0, 0, -1.0])).astype(np.float32),
            "lin_vel": (R.T @ d.qvel[0:3]).astype(np.float32),            # body frame
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
        if self.d.qpos[2] < 0.5:                     # xml gave no standing height
            self.d.qpos[2] = 0.8
        self.d.qpos[self.q_adr] = self.default_q
        self.d.qpos[self.hand_q] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.sim_steps = self.steps = 0
        self.last_action[:] = 0
        return self._obs(), {}

    def step_targets(self, target_q, pre_substep=None):
        """Advance one control step (decimation physics steps) with absolute PD joint targets.
        pre_substep(data), if given, is called before every physics step (e.g. to apply forces)."""
        d, m = self.d, self.m
        kp, kd, lim = self.kp, self.kd, self.tau_lim
        qa, va, aa = self.q_adr, self.v_adr, self.act_adr
        has_hand = len(self.hand_act) > 0
        for _ in range(self.decim):
            tau = kp * (target_q - d.qpos[qa]) - kd * d.qvel[va]
            d.ctrl[aa] = np.clip(tau, -lim, lim)
            if has_hand:
                th = -HAND_KP * d.qpos[self.hand_q] - HAND_KD * d.qvel[self.hand_v]
                d.ctrl[self.hand_act] = np.clip(th, -HAND_TAU, HAND_TAU)
            if pre_substep is not None:
                pre_substep(d)
            mujoco.mj_step(m, d)
        self.sim_steps += self.decim
        self.steps += 1

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -10, 10)
        self.step_targets(self.default_q + self.action_scale * action)
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
