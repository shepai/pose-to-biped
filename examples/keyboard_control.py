"""
Drive the G1 with the keyboard using the pretrained policy through the gym env.

    python keyboard_control.py                 # uses g1.yaml and the policy_path inside it
    python keyboard_control.py --config g1.yaml --policy path/to/motion.pt

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

from g1_loco_env import G1LocoEnv, TorchScriptPolicy

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
    ap.add_argument("--config", default="g1.yaml")
    ap.add_argument("--policy", default=None, help="override the policy path from the yaml")
    args = ap.parse_args()

    flags = {"reset": False}
    env_ref = {}

    def key_cb(keycode):
        env = env_ref.get("env")
        if env is None:
            return
        if keycode in KEYS:
            env.cmd[:] = np.clip(env.cmd + CMD_STEP * np.array(KEYS[keycode], np.float32), CMD_MIN, CMD_MAX)
        elif keycode == glfw.KEY_SPACE:
            env.cmd[:] = 0
        elif keycode == glfw.KEY_BACKSPACE:
            flags["reset"] = True
        else:
            return
        print(f"cmd: vx={env.cmd[0]:+.2f}  vy={env.cmd[1]:+.2f}  yaw={env.cmd[2]:+.2f}")

    env = G1LocoEnv(args.config, render_mode="human", key_callback=key_cb, max_episode_seconds=1e9)
    env_ref["env"] = env
    policy = TorchScriptPolicy(args.policy or env.policy_path)

    obs, _ = env.reset()
    policy.reset()
    env.render()                                   # opens the viewer

    while env.viewer_is_running():
        t0 = time.time()
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        env.render()
        if terminated or truncated or flags["reset"]:
            print("Episode ended - resetting" + (" (fell)" if terminated else ""))
            obs, _ = env.reset()
            policy.reset()
            flags["reset"] = False
        time.sleep(max(0.0, env.dt - (time.time() - t0)))
    env.close()


if __name__ == "__main__":
    main()