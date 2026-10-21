"""
Drive the G1 with the arrow keys (examples/keyboard_control.py).

    python keyboard_control.py --policy path/to/motion.pt

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
import gymnasium as gym
import numpy as np

import G1_tele                                        # noqa: F401  (registers "G1-walker")
from G1_tele.envs.walker import LEG_SLICE
from G1_tele.policies import TorchScriptPolicy, build_policy_obs

N_POLICY_ACTIONS = 12                                 # the pretrained policy drives the 12 leg joints

CMD_STEP = np.array([0.2, 0.2, 0.3], np.float32)
CMD_MIN = np.array([-0.6, -0.4, -1.0], np.float32)
CMD_MAX = np.array([1.0, 0.4, 1.0], np.float32)
KEYS = {
    glfw.KEY_UP: (1, 0, 0), glfw.KEY_DOWN: (-1, 0, 0),
    glfw.KEY_LEFT: (0, 0, 1), glfw.KEY_RIGHT: (0, 0, -1),
    glfw.KEY_COMMA: (0, 1, 0), glfw.KEY_PERIOD: (0, -1, 0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="TorchScript locomotion policy (e.g. motion.pt)")
    args = ap.parse_args()

    cmd = np.zeros(3, np.float32)                     # [vx, vy, yaw_rate]
    flags = {"reset": False}

    def key_cb(keycode):
        if keycode in KEYS:
            cmd[:] = np.clip(cmd + CMD_STEP * np.array(KEYS[keycode], np.float32), CMD_MIN, CMD_MAX)
        elif keycode == glfw.KEY_SPACE:
            cmd[:] = 0
        elif keycode == glfw.KEY_BACKSPACE:
            flags["reset"] = True
        else:
            return
        print(f"cmd: vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  yaw={cmd[2]:+.2f}")

    policy = TorchScriptPolicy(args.policy)
    env = gym.make("G1-walker", render_mode="human", key_callback=key_cb, max_episode_seconds=1e9)
    base = env.unwrapped                              # raw env: state(), dt, n_joints, viewer_is_running()

    env.reset()
    policy.reset()
    last_a = np.zeros(N_POLICY_ACTIONS, np.float32)
    action = np.zeros(base.n_joints, np.float32)
    env.render()                                      # opens the viewer

    try:
        while base.viewer_is_running():
            t0 = time.time()
            last_a = policy(build_policy_obs(base, cmd, last_a))
            action[:] = 0.0
            action[LEG_SLICE] = last_a                # legs from the policy, the rest held at default
            _, _, terminated, truncated, _ = env.step(action)
            env.render()

            if terminated or truncated or flags["reset"]:
                print("Resetting" + (" (robot fell)" if terminated else ""))
                env.reset()
                policy.reset()
                last_a[:] = 0
                cmd[:] = 0
                flags["reset"] = False
            time.sleep(max(0.0, base.dt - (time.time() - t0)))
    finally:
        env.close()


if __name__ == "__main__":
    main()