"""
Drive the G1 with the arrow keys using the pretrained Unitree locomotion policy.

    python keyboard_control.py --policy deploy/pre_train/g1/motion.pt

The env (g1_env.py) loads assets/g1_29dof_with_hand.xml and knows nothing about the policy.
Everything policy-specific lives here: loading the file, building its 47-dim observation,
and mapping its 12 outputs onto the 12 leg joints (arms/waist/hands stay at their default pose).

Keys (click the viewer window first):
    UP / DOWN       forward / backward      (+/- 0.2 m/s per press)
    LEFT / RIGHT    turn left / right       (+/- 0.3 rad/s per press)
    , / .           strafe left / right
    SPACE           stop
    BACKSPACE       reset the episode
"""
import argparse
import time

import glfw
import numpy as np
import torch

from G1_tele.envs.walker import G1Env, LEG_SLICE

# ---- values the pretrained policy was trained with (unitree_rl_gym g1.yaml) ----
CMD_SCALE = np.array([2.0, 2.0, 0.25], np.float32)
PHASE_PERIOD = 0.8
N_POLICY_ACTIONS = 12

CMD_STEP = np.array([0.2, 0.2, 0.3], np.float32)
CMD_MIN = np.array([-0.6, -0.4, -1.0], np.float32)
CMD_MAX = np.array([1.0, 0.4, 1.0], np.float32)
KEYS = {
    glfw.KEY_UP: (1, 0, 0), glfw.KEY_DOWN: (-1, 0, 0),
    glfw.KEY_LEFT: (0, 0, 1), glfw.KEY_RIGHT: (0, 0, -1),
    glfw.KEY_COMMA: (0, 1, 0), glfw.KEY_PERIOD: (0, -1, 0),
}


class TorchScriptPolicy:
    def __init__(self, path):
        self.net = torch.jit.load(str(path), map_location="cpu").eval()

    def reset(self):
        if hasattr(self.net, "reset_memory"):          # recurrent (LSTM) exports
            self.net.reset_memory()

    def __call__(self, obs):
        with torch.no_grad():
            return self.net(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()


def build_policy_obs(env, cmd, last_policy_action):
    s = env.state()
    phase = (s["time"] % PHASE_PERIOD) / PHASE_PERIOD
    return np.concatenate([
        s["ang_vel"] * 0.25,
        s["gravity"],
        cmd * CMD_SCALE,
        s["q"][LEG_SLICE] - env.default_q[LEG_SLICE],
        s["dq"][LEG_SLICE] * 0.05,
        last_policy_action,
        [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
    ]).astype(np.float32)


