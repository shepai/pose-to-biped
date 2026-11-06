"""
examples/run_logging.py  -  everything a training run writes, all under logs/<run>/

    logs/<run>/
        config.json                          env id + kwargs, PPO settings, intervals, start time
        progress.csv                         SB3 training log + track/* and eval/* columns
        eval.csv                             one row per evaluation: survival + errors, overall and per kind
        curriculum.json                      which motion kinds are unlocked, sampling weights
        videos/<run>_005pct_3.0M.mp4         deterministic policy, one segment per motion kind
        checkpoints/<run>_010pct_6.0M.zip    + <same name>_vecnormalize.pkl
        <run>.zip, <run>_vecnormalize.pkl    final model (also written on Ctrl-C)
        events.out.tfevents.*                if tensorboard is installed

Percentages are of this run's total_timesteps, counted from where the run started, so a resumed
stage 2 also gets its own 0..100 %.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


# ------------------------------------------------------------------ OpenGL backend for videos
GL_INFO = {"backend": None, "why": "not configured"}


def configure_mujoco_gl():
    """Choose MuJoCo's off-screen OpenGL backend. Must run BEFORE anything imports mujoco: MuJoCo
    reads MUJOCO_GL once, at import. Its default, glfw, needs a display - over SSH, on a server,
    in a container or in WSL without WSLg it cannot make a renderer and every video fails. With no
    display we switch to EGL (GPU, needs libEGL) - check with `python train_pose_tracking.py video`."""
    if "mujoco" in sys.modules:
        GL_INFO.update(backend=os.environ.get("MUJOCO_GL", "glfw"), why="mujoco was already imported")
    elif os.environ.get("MUJOCO_GL"):
        GL_INFO.update(backend=os.environ["MUJOCO_GL"], why="set in your environment")
    elif sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        os.environ["MUJOCO_GL"] = "egl"
        GL_INFO.update(backend="egl", why="no display found, so EGL was chosen automatically")
    else:
        GL_INFO.update(backend="glfw", why="default (a display is available)")
    return GL_INFO["backend"]


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
    """Write frames (list of HxWx3 uint8) to `path`. Tries imageio (ffmpeg) -> OpenCV -> GIF (Pillow)
    -> raw frames (.npz), and checks each file really exists afterwards - OpenCV without an mp4
    codec "succeeds" without writing anything. Returns (path_written, how); never raises."""
    frames = [np.asarray(f, np.uint8) for f in frames]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = lambda p: os.path.exists(p) and os.path.getsize(p) > 100
    errors = []
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)
        if written(path):
            return path, "mp4 (imageio)"
        errors.append("imageio wrote nothing")
    except Exception as e:                                       # noqa: BLE001
        errors.append(f"imageio: {str(e).splitlines()[0][:120] if str(e) else type(e).__name__}")
    try:
        import cv2
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError("VideoWriter could not open an mp4 file (no codec)")
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        if written(path):
            return path, "mp4 (opencv)"
        errors.append("opencv wrote nothing")
    except Exception as e:                                       # noqa: BLE001
        errors.append(f"opencv: {str(e).splitlines()[0][:120] if str(e) else type(e).__name__}")
    try:
        from PIL import Image
        gif = os.path.splitext(path)[0] + ".gif"
        imgs = [Image.fromarray(f) for f in frames[::2]]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(2000 / fps), loop=0)
        if written(gif):
            return gif, "gif (no mp4 writer: " + "; ".join(errors) + ")"
    except Exception as e:                                       # noqa: BLE001
        errors.append(f"pillow: {e}")
    npz = os.path.splitext(path)[0] + "_frames.npz"
    np.savez_compressed(npz, frames=np.stack(frames[::2]), fps=fps / 2)
    return npz, "raw frames only (" + "; ".join(errors) + ")"


# ------------------------------------------------------------------ callbacks
class TrackingStatsCallback(BaseCallback):
    """Per-rollout metrics (track/* columns) - cleaner than the episode return.
    track/end_fell and track/end_lost split early episode ends into falls and lost tracking."""

    KEYS = {"kp": "kp_err", "hand": "hand_err", "foot": "foot_err", "assist": "assist", "cp": "cp_margin",
            "contact": "contact_match", "r_bal": "r_balance", "r_trk": "r_track",
            # locomotion=True only (logged when present)
            "vel": "vel_err", "wz": "yaw_rate_err", "root": "root_err", "yaw": "yaw_err", "speed": "ref_speed"}

    def __init__(self, dt=0.02):
        super().__init__()
        self.dt = dt

    def _on_rollout_start(self):
        self.buf = {k: [] for k in list(self.KEYS) + ["rew"]}
        self.ep_len, self.fell, self.lost, self.eps = [], 0, 0, 0
        self.off_course = 0

    def _on_step(self):
        infos, dones = self.locals["infos"], self.locals["dones"]
        env = self.training_env
        raw = env.get_original_reward() if hasattr(env, "get_original_reward") else self.locals["rewards"]
        self.buf["rew"].append(float(np.mean(raw)))
        for info, done in zip(infos, dones):
            for k, key in self.KEYS.items():
                self.buf[k].append(info.get(key, np.nan))
            if done:
                self.eps += 1
                self.fell += int(bool(info.get("fell", False)))
                self.lost += int(bool(info.get("lost", False)) and not info.get("fell", False))
                self.off_course += int(bool(info.get("off_course", False)) and not info.get("fell", False))
                if "episode" in info:
                    self.ep_len.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self):
        m = lambda k: float(np.nanmean(self.buf[k])) if len(self.buf[k]) else np.nan
        rec = self.logger.record
        rec("track/kp_err_cm", 100 * m("kp"))
        rec("track/hand_err_cm", 100 * m("hand"))
        rec("track/foot_err_cm", 100 * m("foot"))
        rec("track/reward_per_step", m("rew"))
        rec("track/reward_balance", m("r_bal"))
        rec("track/reward_tracking", m("r_trk"))
        rec("track/cp_margin_cm", 100 * m("cp"))
        rec("track/contact_match", m("contact"))
        rec("track/assist", m("assist"))
        loco = np.isfinite(m("vel"))
        if loco:                                            # locomotion=True runs
            rec("track/vel_err_m_s", m("vel"))
            rec("track/yaw_rate_err", m("wz"))
            rec("track/root_err_cm", 100 * m("root"))
            rec("track/yaw_err_deg", float(np.degrees(m("yaw"))))
            rec("track/ref_speed_m_s", m("speed"))
        if self.eps:
            rec("track/fall_rate", (self.fell + self.lost) / self.eps)
            rec("track/end_fell", self.fell / self.eps)
            rec("track/end_lost", self.lost / self.eps)
            if loco:
                rec("track/end_off_course", self.off_course / self.eps)
        if self.ep_len:
            rec("track/ep_len_s", float(np.mean(self.ep_len)) * self.dt)


class CurriculumCallback(BaseCallback):
    """Unlocks harder motion kinds only once the robot keeps its balance on the easier ones.

    order      kinds from easiest to hardest; kinds not listed (e.g. your own "keypoints") are always on
    start      how many kinds of `order` are unlocked at the start (2 = upper + whole body, feet planted)
    threshold  unlock the next kind when every unlocked kind has mean survival >= threshold
               (survival = fraction of the possible episode the robot stayed up and on track)
    Unlocked kinds are sampled with weight floor + (1 - survival): more practice where it still
    falls, but easy kinds never disappear (so they are not forgotten). State is saved to
    <log_dir>/curriculum.json; stage 2 starts from stage 1's file.
    per_second=True divides each kind's weight by its total clip length, so the weights are shares of
    episodes even when the kinds have different amounts of motion (the env samples clips in
    proportion to their length)."""

    def __init__(self, log_dir, order=("upper", "whole", "balance", "step"), start=2, threshold=0.8,
                 window=400, min_episodes=200, floor=0.3, state_from=None, per_second=False, verbose=1):
        super().__init__(verbose)
        self.order, self.start_level, self.threshold = list(order), start, threshold
        self.window, self.min_eps, self.floor = window, min_episodes, floor
        self.path = os.path.join(log_dir, "curriculum.json")
        self.state_from = state_from
        self.per_second = per_second

    def _on_training_start(self):
        kinds = list(self.training_env.get_attr("clip_kinds", indices=[0])[0])
        self.ladder = [k for k in self.order if k in kinds]
        self.always = [k for k in kinds if k not in self.order]
        self.level = min(self.start_level, len(self.ladder))
        if self.state_from and os.path.exists(self.state_from):
            with open(self.state_from) as f:
                self.level = min(len(self.ladder), max(self.level, int(json.load(f).get("level", 0))))
        self.seconds = dict(self.training_env.get_attr("kind_seconds", indices=[0])[0]) if self.per_second else {}
        self.hist = {k: [] for k in kinds}
        self._push_weights(announce=True)

    def unlocked(self):
        return self.always + self.ladder[:self.level]

    def _survival(self, k):
        h = self.hist[k]
        return float(np.mean(h)) if len(h) >= self.min_eps else None

    def _push_weights(self, announce=False):
        w = {}
        for k in self.unlocked():
            s = self._survival(k)
            w[k] = self.floor + (1.0 - (s if s is not None else 0.0))
        self.weights = w
        if self.per_second:
            w = {k: v / max(self.seconds.get(k, 1.0), 1e-6) for k, v in w.items()}
        self.training_env.env_method("set_kind_weights", w)
        with open(self.path, "w") as f:
            json.dump({"level": self.level, "unlocked": self.unlocked(), "weights": w,
                       "timesteps": int(self.model.num_timesteps)}, f, indent=2)
        if announce and self.verbose:
            print(f"[curriculum] training on: {', '.join(self.unlocked())}")

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "clip_kind" in info and info["clip_kind"] in self.hist:
                h = self.hist[info["clip_kind"]]
                h.append(float(info["survival"]))
                if len(h) > self.window:
                    del h[0]
        return True

    def _on_rollout_end(self):
        surv = [self._survival(k) for k in self.unlocked()]
        if self.level < len(self.ladder) and all(s is not None and s >= self.threshold for s in surv):
            self.level += 1
            if self.verbose:
                print(f"[curriculum] unlocked '{self.ladder[self.level - 1]}' at {self.model.num_timesteps:,} steps")
        self._push_weights()
        self.logger.record("curriculum/level", self.level)
        for k in self.hist:
            s = self._survival(k)
            if s is not None:
                self.logger.record(f"curriculum/survival_{k}", s)
            self.logger.record(f"curriculum/weight_{k}", self.weights.get(k, 0.0))


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
    """Every `every_pct` % (and at 0 %): run the deterministic policy on the same fixed episodes
    (episodes_per_kind for every clip kind) in a separate, unassisted env. The video shows the first
    `video_seconds_per_kind` of one episode per kind, back to back. Results go to eval.csv (overall
    and per kind) and to eval/* columns in progress.csv. Rendering problems never stop training -
    they are reported once and videos are skipped; the evaluation numbers are still logged."""

    def __init__(self, env_id, total_timesteps, log_dir, run_name, every_pct=5, video_seconds_per_kind=4.0,
                 episodes_per_kind=2, max_episode_seconds=10.0, env_kwargs=None, width=640, height=480,
                 distance=2.6, azimuth=135.0, elevation=-15.0):
        super().__init__(total_timesteps, every_pct, at_start=True)
        self.env_id, self.env_kwargs = env_id, dict(env_kwargs or {})
        self.log_dir, self.run = log_dir, run_name
        self.video_s, self.per_kind, self.max_ep_s = video_seconds_per_kind, episodes_per_kind, max_episode_seconds
        self.size, self.cam_cfg = (width, height), (distance, azimuth, elevation)
        self.env = self.renderer = None
        self.render_ok = True
        self.video_status = "not started"
        self.announced = False
        self.t0 = time.time()
        self.csv_path = os.path.join(log_dir, "eval.csv")

    # ---------- setup
    def _setup(self):
        if self.env is not None:
            return
        self.env = gym.make(self.env_id, **self.env_kwargs).unwrapped
        lib = self.env.lib
        rng = np.random.default_rng(1234)
        self.kinds = list(self.env.clip_kinds)
        self.episodes = []                                       # fixed (kind, clip, t0) list
        for k in self.kinds:
            clips = [c for c in range(lib.n_clips) if str(lib.kinds[c]) == k and lib.duration(c) >= 2.0]
            for c in [clips[int(i)] for i in np.linspace(0, len(clips) - 1, self.per_kind)] if clips else []:
                t0 = float(rng.uniform(0, max(0.0, lib.duration(c) - self.max_ep_s)))
                self.episodes.append((k, c, t0))
        self.loco = bool(getattr(self.env, "locomotion", False))
        self.fields = (["pct", "timesteps", "wall_minutes", "survival", "ep_len_s", "kp_err_cm", "hand_err_cm",
                        "foot_err_cm"] + (["vel_err_m_s", "root_err_cm"] if self.loco else [])
                       + [f"survival_{k}" for k in self.kinds]
                       + [f"kp_err_cm_{k}" for k in self.kinds]
                       + ([f"vel_err_m_s_{k}" for k in self.kinds] if self.loco else []) + ["video"])
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
            self._videos_off(f"could not create an off-screen renderer: {str(e).strip() or type(e).__name__}")

    def _videos_off(self, reason):
        self.video_status = "off"
        gl = os.environ.get("MUJOCO_GL", "glfw (default)")
        bar = "=" * 92
        print(f"\n{bar}\n[video] VIDEOS ARE OFF - {reason}\n"
              f"[video] MUJOCO_GL={gl} ({GL_INFO['why']})\n"
              f"[video] Training and the eval.csv numbers carry on. To find a backend that works, run\n"
              f"[video]     python train_pose_tracking.py video\n"
              f"[video] then start training with it, e.g.  MUJOCO_GL=egl python train_pose_tracking.py train\n{bar}\n",
              flush=True)

    def _frame(self, frames):
        try:
            self.renderer.update_scene(self.env.data, camera=self.cam)
            self.env.draw_targets(self.renderer.scene)
            frames.append(self.renderer.render().copy())
        except Exception as e:                                   # noqa: BLE001
            self.renderer, self.render_ok = None, False
            self._videos_off(f"rendering a frame failed: {str(e).strip() or type(e).__name__}")

    # ---------- one evaluation
    def _fire(self, pct):
        self._setup()
        env, model = self.env, self.model
        vn = model.get_vec_normalize_env()
        n_video = int(self.video_s / env.dt)
        max_steps = int(self.max_ep_s / env.dt)
        frames, filmed = [], set()
        res = {k: {"surv": [], "len": [], "kp": [], "hand": [], "foot": [], "vel": [], "root": []} for k in self.kinds}
        for i, (kind, clip, t0) in enumerate(self.episodes):
            obs, _ = env.reset(seed=10_000 + i, options={"clip": clip, "t0": t0})
            film = kind not in filmed and self.renderer is not None
            filmed.add(kind)
            r = res[kind]
            for k in range(max_steps):
                o = vn.normalize_obs(obs) if vn is not None else obs
                action, _ = model.predict(o, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                r["kp"].append(info["kp_err"]); r["hand"].append(info["hand_err"]); r["foot"].append(info["foot_err"])
                if self.loco:
                    r["vel"].append(info["vel_err"]); r["root"].append(info["root_err"])
                if film and k < n_video and self.renderer is not None:
                    self._frame(frames)
                if term or trunc:
                    break
            if film and self.renderer is not None and frames:            # hold the last frame briefly
                frames += [frames[-1]] * max(0, min(10, n_video - (k + 1)))
            r["surv"].append(info.get("survival", 1.0))
            r["len"].append((k + 1) * env.dt)

        label = _pct_label(pct, model.num_timesteps)
        video, video_col = "", "off (see training log)" if not self.render_ok else "no frames"
        if frames:
            if max(int(f.max()) for f in frames[:: max(1, len(frames) // 10)]) == 0:
                print("[video] warning: rendered frames are completely black (broken OpenGL driver?)")
            video, how = write_video(frames, os.path.join(self.log_dir, "videos", f"{self.run}_{label}.mp4"),
                                     fps=round(1.0 / env.dt))
            video_col = os.path.basename(video)
            if not self.announced or not how.startswith("mp4"):
                print(f"[video] {how}: {video}  (MUJOCO_GL={os.environ.get('MUJOCO_GL', 'glfw (default)')})")
                self.announced = True
            self.video_status = "on"
        cat = lambda key: [v for k in self.kinds for v in res[k][key]]
        cm = lambda xs: round(100 * float(np.nanmean(xs)), 2) if len(xs) else np.nan
        row = {"pct": round(pct, 1), "timesteps": int(model.num_timesteps),
               "wall_minutes": round((time.time() - self.t0) / 60, 1),
               "survival": round(float(np.mean(cat("surv"))), 3), "ep_len_s": round(float(np.mean(cat("len"))), 2),
               "kp_err_cm": cm(cat("kp")), "hand_err_cm": cm(cat("hand")), "foot_err_cm": cm(cat("foot")),
               "video": video_col}
        for k in self.kinds:
            row[f"survival_{k}"] = round(float(np.mean(res[k]["surv"])), 3) if res[k]["surv"] else np.nan
            row[f"kp_err_cm_{k}"] = cm(res[k]["kp"])
        if self.loco:
            mean = lambda xs: round(float(np.nanmean(xs)), 3) if len(xs) else np.nan   # noqa: E731
            row["vel_err_m_s"], row["root_err_cm"] = mean(cat("vel")), cm(cat("root"))
            for k in self.kinds:
                row[f"vel_err_m_s_{k}"] = mean(res[k]["vel"])
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=self.fields)
            if new:
                wr.writeheader()
            wr.writerow(row)
        for key in ["survival", "ep_len_s", "kp_err_cm", "hand_err_cm", "foot_err_cm"] + \
                   (["vel_err_m_s", "root_err_cm"] if self.loco else []) + [f"survival_{k}" for k in self.kinds]:
            self.logger.record(f"eval/{key}", row[key])
        per_kind = "  ".join(f"{k} {row[f'survival_{k}']:.2f}" for k in self.kinds)
        walk = f", speed err {row['vel_err_m_s']:.2f} m/s" if self.loco else ""
        print(f"[eval {row['pct']:5.1f}%] survival {row['survival']:.2f} ({per_kind}), "
              f"kp err {row['kp_err_cm']:.1f} cm{walk}, video: {video_col}", flush=True)

    def record_now(self, label_pct=None):
        """Extra evaluation outside the schedule (used for the final model, e.g. after Ctrl-C)."""
        pct = label_pct if label_pct is not None else 100.0 * self._steps_done() / max(1, self.total)
        self._fire(pct)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

