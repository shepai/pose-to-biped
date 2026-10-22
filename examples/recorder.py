import os 
import cv2
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import mujoco
def write_video(frames, path, fps=30):
    if not frames:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")

    try:
        for frame in frames:
            # Gymnasium returns RGB; OpenCV expects BGR.
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

class TrainingVideoCallback(BaseCallback):
    def __init__(
        self,
        env_name,
        total_timesteps,
        video_dir,
        every_percent=5,
        video_seconds=15,
        verbose=1,
    ):
        super().__init__(verbose)

        self.env_name = env_name
        self.total_timesteps = total_timesteps
        self.video_dir = video_dir
        self.every_percent = every_percent
        self.video_seconds = video_seconds

        self.next_percent = every_percent
        self.render_env = None

    def _on_step(self):
        progress = 100 * self.num_timesteps / self.total_timesteps

        while (
            self.next_percent <= 100
            and progress >= self.next_percent
        ):
            self._record(self.next_percent)
            self.next_percent += self.every_percent

        return True
    def _on_training_start(self):
        os.makedirs(self.video_dir, exist_ok=True)

        self.render_env = gym.make(self.env_name)              # no render_mode
        raw = self.render_env.unwrapped
        self.renderer = mujoco.Renderer(raw.model, height=480, width=640)

        core = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, "core")
        self.cam = mujoco.MjvCamera()                          # follows the robot
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = core
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.5, 135, -25

    def _record(self, percent):
        env = self.render_env
        raw = env.unwrapped
        obs, _ = env.reset()

        policy_dt = raw.frame_skip * raw.model.opt.timestep
        fps = max(1, round(1.0 / policy_dt))
        n_frames = max(1, round(self.video_seconds * fps))
        frames = []

        for _ in range(n_frames):
            self.renderer.update_scene(raw.data, camera=self.cam)
            frames.append(self.renderer.render().copy())

            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        path = os.path.join(self.video_dir,
                            f"progress_{percent:03d}pct_{self.num_timesteps}.mp4")
        write_video(frames, path, fps=fps)
        if self.verbose:
            print(f"Saved training video: {path}")

    def _on_training_end(self):
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None
        if self.render_env is not None:
            self.render_env.close()
            self.render_env = None