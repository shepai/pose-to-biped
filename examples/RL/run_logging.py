"""
examples/run_logging.py  -  everything a training run writes, all under logs/<run>/

    logs/<run>/
        config.json                          env id + kwargs, PPO settings, intervals, start time
        progress.csv                         SB3 training log + track/* and eval/* columns
        eval.csv                             one row per evaluation (every video_every_pct %)
        videos/<run>_005pct_3.0M.mp4         deterministic policy, target spheres drawn
        checkpoints/<run>_010pct_6.0M.zip    + <same name>_vecnormalize.pkl
        <run>.zip, <run>_vecnormalize.pkl    final model (also written on Ctrl-C)
        events.out.tfevents.*                if tensorboard is installed

Percentages are of this run's total_timesteps, counted from where the run started, so a resumed
stage 2 also gets its own 0..100 %.
"""
import csv
import json
import os
import time

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


# ------------------------------------------------------------------ run directory / config
def prepare_run_dir(log_dir):
    """Never overwrite an old run: if logs/<run> already has files, rename it logs/<run>_<date-time>."""
    if os.path.isdir(log_dir) and os.listdir(log_dir):
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(log_dir)))
        old = f"{log_dir}_{stamp}"
        os.rename(log_dir, old)
        print(f"[logs] previous run moved to {old}")
    os.makedirs(os.path.join(log_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    return log_dir


def save_config(log_dir, **cfg):
    cfg["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def _pct_label(pct, steps):
    return f"{int(round(pct)):03d}pct_{steps / 1e6:.1f}M"


# ------------------------------------------------------------------ video writing
def write_video(frames, path, fps):
    """mp4 via imageio or OpenCV; falls back to a GIF (Pillow) if neither is installed.
    Returns the path actually written."""
    frames = [np.asarray(f, np.uint8) for f in frames]
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)
        return path
    except Exception:                                            # noqa: BLE001
        pass
    try:
        import cv2
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return path
    except Exception:                                            # noqa: BLE001
        pass
    from PIL import Image
    gif = os.path.splitext(path)[0] + ".gif"
    imgs = [Image.fromarray(f) for f in frames[::2]]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(2000 / fps), loop=0)
    print("[video] imageio-ffmpeg / opencv not found - wrote a GIF instead "
          "(pip install imageio imageio-ffmpeg for mp4)")
    return gif


# ------------------------------------------------------------------ callbacks
class TrackingStatsCallback(BaseCallback):
    """Per-rollout tracking metrics (track/* columns) - cleaner than the episode return."""

    def __init__(self, dt=0.02):
        super().__init__()
        self.dt = dt

    def _on_rollout_start(self):
        self.buf = {"kp": [], "hand": [], "foot": [], "rew": [], "assist": []}
        self.ep_len, self.falls, self.eps = [], 0, 0

    def _on_step(self):
        infos, dones = self.locals["infos"], self.locals["dones"]
        env = self.training_env
        raw = env.get_original_reward() if hasattr(env, "get_original_reward") else self.locals["rewards"]
        self.buf["rew"].append(float(np.mean(raw)))
        for info, done in zip(infos, dones):
            self.buf["kp"].append(info.get("kp_err", np.nan))
            self.buf["hand"].append(info.get("hand_err", np.nan))
            self.buf["foot"].append(info.get("foot_err", np.nan))
            self.buf["assist"].append(info.get("assist", 0.0))
            if done:
                self.eps += 1
                self.falls += int(info.get("fell", False) or info.get("lost", False))
                if "episode" in info:
                    self.ep_len.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self):
        self.logger.record("track/kp_err_cm", 100 * np.nanmean(self.buf["kp"]))
        self.logger.record("track/hand_err_cm", 100 * np.nanmean(self.buf["hand"]))
        self.logger.record("track/foot_err_cm", 100 * np.nanmean(self.buf["foot"]))
        self.logger.record("track/reward_per_step", float(np.mean(self.buf["rew"])))
        self.logger.record("track/assist", float(np.mean(self.buf["assist"])))
        if self.eps:
            self.logger.record("track/fall_rate", self.falls / self.eps)
        if self.ep_len:
            self.logger.record("track/ep_len_s", float(np.mean(self.ep_len)) * self.dt)


class _PercentCallback(BaseCallback):
    """Fires self._fire(pct) at 0, every_pct, 2*every_pct, ... % of this run's total_timesteps."""

    def __init__(self, total_timesteps, every_pct, at_start=True):
        super().__init__()
        self.total, self.every = int(total_timesteps), float(every_pct)
        self.at_start = at_start

    def _on_training_start(self):
        self.start = int(self.model.num_timesteps)
        self.next_pct = 0.0 if self.at_start else self.every
        self._maybe_fire()

    def _steps_done(self):
        return int(self.model.num_timesteps) - self.start

    def _maybe_fire(self):
        if self.every <= 0:
            return
        pct = 100.0 * self._steps_done() / max(1, self.total)
        while pct + 1e-9 >= self.next_pct and self.next_pct <= 100.0 + 1e-9:
            self._fire(self.next_pct)
            self.next_pct += self.every

    def _on_step(self):
        self._maybe_fire()
        return True

    def _fire(self, pct):
        raise NotImplementedError


class PeriodicCheckpointCallback(_PercentCallback):
    """Backs up model + VecNormalize stats to logs/<run>/checkpoints every `every_pct` %."""

    def __init__(self, total_timesteps, log_dir, run_name, every_pct=10):
        super().__init__(total_timesteps, every_pct, at_start=False)
        self.dir, self.run = os.path.join(log_dir, "checkpoints"), run_name

    def _fire(self, pct):
        name = f"{self.run}_{_pct_label(pct, self.model.num_timesteps)}"
        self.model.save(os.path.join(self.dir, name))
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(os.path.join(self.dir, f"{name}_vecnormalize.pkl"))
        if self.verbose:
            print(f"[checkpoint] {name}")


class PeriodicVideoEvalCallback(_PercentCallback):
    """Every `every_pct` % (and at 0 %): run the deterministic policy on a fixed set of episodes in a
    separate, unassisted env; film the first `video_seconds`; write the video, a row in eval.csv and
    eval/* columns in progress.csv. Rendering problems never stop training - they are reported once
    and videos are skipped, the evaluation numbers are still logged."""

    FIELDS = ["pct", "timesteps", "wall_minutes", "episodes", "ep_len_s", "survived_full_frac",
              "kp_err_cm", "hand_err_cm", "foot_err_cm", "video"]

    def __init__(self, env_id, total_timesteps, log_dir, run_name, every_pct=5, video_seconds=8.0,
                 eval_episodes=5, max_episode_seconds=10.0, env_kwargs=None, width=640, height=480,
                 distance=2.6, azimuth=135.0, elevation=-15.0):
        super().__init__(total_timesteps, every_pct, at_start=True)
        self.env_id, self.env_kwargs = env_id, dict(env_kwargs or {})
        self.log_dir, self.run = log_dir, run_name
        self.video_seconds, self.eval_episodes, self.max_ep_s = video_seconds, eval_episodes, max_episode_seconds
        self.size, self.cam_cfg = (width, height), (distance, azimuth, elevation)
        self.env = self.renderer = None
        self.render_ok = True
        self.t0 = time.time()
        self.csv_path = os.path.join(log_dir, "eval.csv")

    # ---------- setup
    def _setup(self):
        if self.env is not None:
            return
        self.env = gym.make(self.env_id, **self.env_kwargs).unwrapped
        if not self.render_ok:
            return
        try:
            import mujoco
            m = self.env.model
            w = min(self.size[0], int(m.vis.global_.offwidth))
            h = min(self.size[1], int(m.vis.global_.offheight))
            self.renderer = mujoco.Renderer(m, height=h, width=w)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.env.pelvis_id
            cam.distance, cam.azimuth, cam.elevation = self.cam_cfg
            self.cam = cam
        except Exception as e:                                   # noqa: BLE001
            self.render_ok = False
            print(f"[video] rendering unavailable ({e}); evaluations continue without videos. "
                  "On a headless machine try MUJOCO_GL=egl (or osmesa).")

    # ---------- one evaluation
    def _fire(self, pct):
        self._setup()
        env, model = self.env, self.model
        vn = model.get_vec_normalize_env()
        n_video = int(self.video_seconds / env.dt)
        max_steps = int(self.max_ep_s / env.dt)
        frames, lens, full, kp, hand, foot = [], [], 0, [], [], []
        for ep in range(self.eval_episodes):
            obs, _ = env.reset(seed=10_000 + ep)                 # same episodes every time -> comparable
            for k in range(max_steps):
                o = vn.normalize_obs(obs) if vn is not None else obs
                action, _ = model.predict(o, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                kp.append(info["kp_err"]); hand.append(info["hand_err"]); foot.append(info["foot_err"])
                if self.renderer is not None and len(frames) < n_video:
                    try:
                        self.renderer.update_scene(env.data, camera=self.cam)
                        env.draw_targets(self.renderer.scene)
                        frames.append(self.renderer.render().copy())
                    except Exception as e:                       # noqa: BLE001
                        print(f"[video] render failed ({e}); disabling videos")
                        self.renderer, self.render_ok = None, False
                if term or trunc:
                    full += int(trunc and not term)
                    break
            else:
                full += 1
            lens.append((k + 1) * env.dt)

        label = _pct_label(pct, model.num_timesteps)
        video = ""
        if frames:
            video = write_video(frames, os.path.join(self.log_dir, "videos", f"{self.run}_{label}.mp4"),
                                fps=round(1.0 / env.dt))
        row = {"pct": round(pct, 1), "timesteps": int(model.num_timesteps),
               "wall_minutes": round((time.time() - self.t0) / 60, 1), "episodes": self.eval_episodes,
               "ep_len_s": round(float(np.mean(lens)), 2), "survived_full_frac": round(full / self.eval_episodes, 2),
               "kp_err_cm": round(100 * float(np.nanmean(kp)), 2), "hand_err_cm": round(100 * float(np.nanmean(hand)), 2),
               "foot_err_cm": round(100 * float(np.nanmean(foot)), 2), "video": os.path.basename(video)}
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                wr.writeheader()
            wr.writerow(row)
        for key in ("ep_len_s", "survived_full_frac", "kp_err_cm", "hand_err_cm", "foot_err_cm"):
            self.logger.record(f"eval/{key}", row[key])
        print(f"[eval {row['pct']:5.1f}%] ep_len {row['ep_len_s']:.2f} s, kp err {row['kp_err_cm']:.1f} cm"
              + (f", video {row['video']}" if video else ""))

    def record_now(self, label_pct=None):
        """Extra evaluation outside the schedule (used for the final model, e.g. after Ctrl-C)."""
        pct = label_pct if label_pct is not None else 100.0 * self._steps_done() / max(1, self.total)
        self._fire(pct)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
