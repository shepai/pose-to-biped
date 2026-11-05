"""
examples/RL/live_pose.py  -  the simulated G1 copies the person in front of your webcam

    python examples/RL/live_pose.py                          # webcam 0, the final stage 1 policy
    python examples/RL/live_pose.py --camera 1               # another camera
    python examples/RL/live_pose.py --video clip.mp4 --loop  # a video file instead of a webcam
    python examples/RL/live_pose.py --model logs/pose_stage1/checkpoints/pose_stage1_030pct_18.0M
    python examples/RL/live_pose.py --record live.mp4        # record from the start (or press v any time)
    python examples/RL/live_pose.py --record-view robot      # what recordings contain: window, panels, robot, camera

MediaPipe Pose finds the person in each camera frame. Once their torso (shoulders and hips) has been
visible for a few frames, the simulated robot is reset into their pose and the trained policy starts
following them, while keeping its balance. If they leave, the robot holds the last pose for a moment,
then goes back to standing; the next person found starts a fresh reset. A fall also resets the
robot into the person's current pose. Arms or legs that are out of view (e.g. legs under a desk)
are kept in the robot's default pose.

Stand or sit upright when you first step in: "up" is taken from your torso at that moment, which
corrects a tilted camera. Press c to redo it.

Keys (click the window first):
    v        start / stop recording a video (saved to recordings/ unless --record names the file)
    r        reset the robot into your pose now
    c        re-level the camera from your torso (stand or sit upright)
    m        mirror view on/off
    space    pause / resume the simulation
    q, Esc   quit

Needs: pip install mediapipe==0.10.14 (see the README). Runs on the CPU; a display is needed for the window.
"""
import argparse
import datetime
import os
import pickle
import threading
import time
from pathlib import Path

import cv2
import numpy as np

MODEL_IN_REPO = Path("logs") / "pose_stage1T1" / "pose_stage1T1"      # the final stage 1 model


def find_default_model():
    """The final stage 1 model: looked for in the current folder, in the repository the G1_tele
    package is installed from, and in any folder above this script with a pyproject.toml."""
    roots = [Path.cwd()]
    try:
        import G1_tele
        roots.append(Path(G1_tele.__file__).resolve().parents[2])        # <repo>/src/G1_tele/__init__.py
    except Exception:                                                     # noqa: BLE001
        pass
    roots += [d for d in Path(__file__).resolve().parents if (d / "pyproject.toml").exists()]
    for root in roots:
        if (root / MODEL_IN_REPO).with_name("pose_stage1T1.zip").exists():
            return root / MODEL_IN_REPO
    return MODEL_IN_REPO                                                  # load_policy() reports what's missing


# ------------------------------------------------------------------ policy
def load_policy(model_path):
    """SB3 model + its VecNormalize statistics -> function obs -> action (deterministic)."""
    base = str(model_path)[:-4] if str(model_path).endswith(".zip") else str(model_path)
    stats = base + "_vecnormalize.pkl"
    missing = [f for f in (base + ".zip", stats) if not os.path.exists(f) and not (f == base + ".zip" and os.path.exists(base))]
    if missing:
        raise SystemExit("can't find " + " or ".join(os.path.abspath(f) for f in missing) +
                         "\npass the model explicitly, e.g. --model logs/pose_stage1T1/pose_stage1T1 "
                         "(a model and its _vecnormalize.pkl must sit in the same folder)")
    from stable_baselines3 import PPO
    model = PPO.load(base, device="cpu")
    with open(stats, "rb") as f:
        vecnorm = pickle.load(f)                             # only its observation statistics are used
    vecnorm.training = False

    def act(obs):
        action, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        return action
    return act


# ------------------------------------------------------------------ camera + MediaPipe, in a thread
class PoseCamera(threading.Thread):
    """Reads frames and runs MediaPipe Pose as fast as the camera allows; the newest result is in
    .latest: dict(n, frame (BGR, skeleton drawn), world (33,3) or None, vis (33,) or None, t (s))."""

    def __init__(self, source, loop=False, complexity=1, width=None, height=None):
        super().__init__(daemon=True)
        self.is_file = isinstance(source, str)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open {'video ' + source if self.is_file else f'camera {source}'}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.loop, self.complexity = loop, complexity
        self.latest, self.fps, self.ended, self.error = None, 0.0, False, None
        self._stop = threading.Event()

    def run(self):
        try:
            import mediapipe as mp
            mp_pose, draw = mp.solutions.pose, mp.solutions.drawing_utils
            pose = mp_pose.Pose(model_complexity=self.complexity, smooth_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except Exception as e:                               # noqa: BLE001
            self.error = f"MediaPipe failed to start: {e}"
            return
        n, t_prev, t_next = 0, time.perf_counter(), time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file and self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.ended = True
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            world = vis = None
            if res.pose_world_landmarks is not None and res.pose_landmarks is not None:
                world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks.landmark])
                vis = np.array([p.visibility for p in res.pose_landmarks.landmark])
                draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            n += 1
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            self.latest = {"n": n, "frame": frame, "world": world, "vis": vis, "t": now}
            if self.is_file:                                 # play files at their own speed
                t_next += 1.0 / self.file_fps
                time.sleep(max(0.0, t_next - time.perf_counter()))
        pose.close()

    def stop(self):
        self._stop.set()
        self.join(timeout=2.0)
        self.cap.release()


# ------------------------------------------------------------------ robot view
class RobotView:
    """Off-screen render of the robot from the front, with the target keypoints drawn as spheres."""

    def __init__(self, env, height=480, width=480):
        import mujoco
        self.mujoco, self.env = mujoco, env
        m = env.model
        w, h = min(width, int(m.vis.global_.offwidth)), min(height, int(m.vis.global_.offheight))
        self.renderer = mujoco.Renderer(m, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = env.pelvis_id
        self.cam.distance, self.cam.elevation = 2.8, -12.0

    def render(self):
        from G1_tele.motion.kinematics import yaw_of_quat
        env = self.env
        self.cam.azimuth = np.degrees(yaw_of_quat(env.data.qpos[3:7])) + 180.0     # always face the robot
        self.renderer.update_scene(env.data, camera=self.cam)
        env.draw_targets(self.renderer.scene)
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def close(self):
        self.renderer.close()


def panels(cam_frame, robot, mirror):
    """Camera and robot images at the same height, mirrored if asked: (camera, robot)."""
    h = robot.shape[0]
    if mirror:
        robot = robot[:, ::-1]
    if cam_frame is None:
        cam = np.zeros((h, int(h * 4 / 3), 3), np.uint8)
        cv2.putText(cam, "waiting for the camera...", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        cam = cv2.resize(cam_frame, (int(cam_frame.shape[1] * h / cam_frame.shape[0]), h))
        if mirror:
            cam = cam[:, ::-1]
    return np.ascontiguousarray(cam), np.ascontiguousarray(robot)


def status_bar(lines, width):
    bar = np.zeros((26 * len(lines) + 10, width, 3), np.uint8)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(bar, text, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    return bar


class Recorder:
    """Writes frames to an .mp4 at a steady frame rate: frames are repeated when the window draws more
    slowly, so the video always plays at real speed. The frame size is fixed by the first frame."""

    def __init__(self, path, fps=30):
        self.path, self.fps = str(path), int(fps)
        self.writer, self.size, self.t0, self.n = None, None, None, 0

    def add(self, img, now):
        if self.writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self.size = (img.shape[1] // 2 * 2, img.shape[0] // 2 * 2)          # even sizes suit every codec
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self.writer.isOpened():
                raise SystemExit(f"could not open {self.path} for writing")
            self.t0 = now
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)
        due = int((now - self.t0) * self.fps) + 1 - self.n
        for _ in range(max(0, min(due, 2 * self.fps))):      # catch up, but not after a long stall
            self.writer.write(img)
            self.n += 1

    @property
    def seconds(self):
        return self.n / self.fps

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f"saved {self.path} ({self.seconds:.1f} s)")
        else:
            print(f"nothing recorded to {self.path} (the camera hadn't started)")
        self.writer = None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="trained model, with or without .zip (default: logs/pose_stage1T1/pose_stage1T1)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--video", default=None, help="video file instead of a webcam")
    ap.add_argument("--loop", action="store_true", help="loop the video file")
    ap.add_argument("--cam-width", type=int, default=None, help="ask the webcam for this frame width")
    ap.add_argument("--cam-height", type=int, default=None, help="ask the webcam for this frame height")
    ap.add_argument("--complexity", type=int, default=1, choices=(0, 1, 2),
                    help="MediaPipe model: 0 fastest, 2 most accurate")
    ap.add_argument("--min-visibility", type=float, default=0.5,
                    help="how sure MediaPipe must be of a landmark to use it (0-1)")
    ap.add_argument("--acquire", type=int, default=8, help="camera frames with a torso before the robot resets to you")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the pose after losing you")
    ap.add_argument("--smooth", type=float, default=0.08, help="reference smoothing time constant (s)")
    ap.add_argument("--no-level", action="store_true", help="assume the camera is level (no torso calibration)")
    ap.add_argument("--no-filter", action="store_true", help="turn off the One-Euro keypoint filter")
    ap.add_argument("--com-weight", type=float, default=0.0,
                    help="retargeting: pull the reference's centre of mass over the feet (0 = off)")
    ap.add_argument("--size", type=int, default=480, help="height of the window panels in pixels")
    ap.add_argument("--record", default=None, help="start recording at once, to this .mp4 (or press v any time)")
    ap.add_argument("--record-view", default="window", choices=("window", "panels", "robot", "camera"),
                    help="what a recording shows: the whole window, both panels without the status text, "
                         "or one panel")
    ap.add_argument("--record-dir", default="recordings", help="folder for recordings started with v")
    ap.add_argument("--record-fps", type=int, default=30, help="frame rate of recordings")
    ap.add_argument("--no-mirror", action="store_true", help="show the camera unmirrored")
    a = ap.parse_args()

    import G1_tele                                                     # noqa: F401
    from G1_tele.envs.live import LivePoseTrackingEnv
    from G1_tele.teleop import Teleop

    model = a.model or find_default_model()
    print(f"loading {os.path.abspath(model)}")
    policy = load_policy(model)
    env = LivePoseTrackingEnv()
    teleop = Teleop(env, policy, min_visibility=a.min_visibility, acquire_frames=a.acquire, hold_seconds=a.hold,
                    tau=a.smooth, level=not a.no_level, com_weight=a.com_weight, kin=env.kin,
                    one_euro=not a.no_filter)
    camera = PoseCamera(a.video if a.video else a.camera, loop=a.loop, complexity=a.complexity,
                        width=a.cam_width, height=a.cam_height)
    camera.start()
    view = RobotView(env, height=a.size, width=a.size)
    mirror, paused = not a.no_mirror, False
    recorder = Recorder(a.record, a.record_fps) if a.record else None

    def new_recording():
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return Recorder(os.path.join(a.record_dir, f"live_{stamp}.mp4"), a.record_fps)
    if recorder:
        print(f"recording to {recorder.path}")
    print("running - click the window, then: v record, r reset to you, c re-level, m mirror, space pause, q quit")

    dt = env.dt
    t0, steps, last_n, last_draw = time.perf_counter(), 0, 0, 0.0
    sim_rate, rate_t, rate_steps = 1.0, time.perf_counter(), 0
    try:
        while True:
            if camera.error:
                raise SystemExit(camera.error)
            det = camera.latest
            if det is not None and det["n"] != last_n:                 # a new camera frame
                last_n = det["n"]
                teleop.on_pose(det["world"], det["vis"], det["t"])

            now = time.perf_counter()
            if paused:
                t0 = now - steps * dt
            else:
                due = int((now - t0) / dt) - steps
                if due > 5:                                            # can't keep up: run slower than real time
                    t0 += (due - 5) * dt
                    due = 5
                for _ in range(due):
                    teleop.step()
                    steps += 1
                if due == 5:
                    time.sleep(0.001)                                  # let the camera thread run
            if now - rate_t > 1.0:
                sim_rate, rate_t, rate_steps = (steps - rate_steps) * dt / (now - rate_t), now, steps

            if now - last_draw < 1 / 30:
                time.sleep(0.002)
                continue
            last_draw = now
            info = teleop.last_info
            tracked = teleop.human.tracked
            limbs = ", ".join(n.replace("_", " ") for n, ok in tracked.items() if ok) or "-"
            state = teleop.state + ("  (paused)" if paused else "")
            colour = {"tracking": (80, 220, 80), "holding": (60, 200, 255)}.get(teleop.state, (200, 200, 200))
            lines = [(f"{state.upper()}   resets {teleop.resets}   falls {teleop.falls}   "
                      f"pose error {100 * info.get('kp_err', np.nan):.1f} cm   "
                      f"camera {camera.fps:.0f} fps   sim {sim_rate:.2f}x real time", colour),
                     (f"copying: {limbs}      v record   r reset   c level   m mirror   space pause   q quit",
                      (200, 200, 200))]
            cam_img, robot_img = panels(det["frame"] if det else None, view.render(), mirror)
            both = np.hstack([cam_img, robot_img])
            img = np.vstack([status_bar(lines, both.shape[1]), both])
            if recorder is not None:
                if det is not None:                                    # start once the camera is running
                    recorder.add({"window": img, "panels": both, "robot": robot_img, "camera": cam_img}[a.record_view],
                                 now)
                rec = f"REC {int(recorder.seconds) // 60:02d}:{int(recorder.seconds) % 60:02d}"   # shown, not recorded
                cv2.circle(img, (img.shape[1] - 118, 18), 7, (40, 40, 230), -1)
                cv2.putText(img, rec, (img.shape[1] - 104, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 230), 2, cv2.LINE_AA)
            cv2.imshow("pose to humanoid", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("v"):
                if recorder is None:
                    recorder = new_recording()
                    print(f"recording to {recorder.path} - press v again to stop")
                else:
                    recorder.close()
                    recorder = None
            elif key == ord("r"):
                print("reset to you" if teleop.reset_to_person() else "nobody in view: back to standing")
            elif key == ord("c"):
                deg = teleop.recalibrate()
                print("nobody in view" if deg is None else f"camera levelled ({deg:.1f} degrees)")
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord(" "):
                paused = not paused
            if camera.ended:
                print("end of video")
                break
    finally:
        camera.stop()
        view.close()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()