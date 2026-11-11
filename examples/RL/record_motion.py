"""
examples/RL/record_motion.py  -  record your own motions with a webcam, as training data, unattended

    python examples/RL/record_motion.py --name lab_day1                 # -> src/G1_tele/assets/motions/human/lab_day1/
    python examples/RL/record_motion.py --name desk --parts upper       # upper body only, legs kept standing
    python examples/RL/record_motion.py --name lab --no-window --stop-after 60   # no display, stop after an hour
    python examples/RL/record_motion.py --name dance --video dance.mp4  # a video file instead of the webcam
    python examples/RL/record_motion.py --reprocess src/G1_tele/assets/motions/human/lab_day1   # clean again
    python examples/RL/record_motion.py --name walks --walk             # keep where you walk and turn too
    python examples/RL/record_motion.py --name lab_day1 --robot h1      # retarget onto another robot
                                                  # -> src/G1_tele/assets/motions/h1/human/lab_day1/

It records by itself. Whenever every body point it needs is visible, the frames go into the current
"slot". When the person is lost (a needed point missing, or nobody found) for longer than --max-gap
(0.15 s), the slot ends: if it lasted at least --min-seconds (2 s) it is cleaned and saved, otherwise
it is thrown away as noise. Slots longer than --max-slot (60 s) are saved in pieces, so a long session
is written as it goes. Cleaning and saving run in the background, so recording never pauses.

Only poses are saved - no images or video:
  slot_0001.npz      robot poses (pelvis position and orientation + 29 joint angles, 50 Hz, kind "human"),
                     the training data. A slot whose frames were all rejected while cleaning has none.
  slot_0001_raw.npz  the MediaPipe body points of the slot (33 points with a confidence each, per frame,
                     in metres and in the image), so it can be cleaned again with other settings
                     (--reprocess, also --reprocess ... --walk) without recording again. --no-raw skips these.
Running again with the same --name adds new slots to the folder. For training:
    python examples/RL/finetune.py --from ... --name ... --motions src/G1_tele/assets/motions/human/lab_day1

What is never saved (see src/G1_tele/motion/recording.py): frames where a needed body point is missing
or unsure, glitch frames (MediaPipe throwing a point away for a frame or two), poses the robot can't
reach and joints moving implausibly fast. A clip never spans a gap, so saved motions have no jumps.
Camera tilt is corrected from the person's average posture over the session (people are upright most of
the time), so nobody has to stand still for a calibration.

--walk: the clips also keep where you walked and which way you faced (kind "human_walk" unless --kind
is given), for training walking policies:
    python examples/RL/train_locomotion.py --from ... --name ... --motions src/G1_tele/assets/motions/human/walks
Keep your whole body in view, and give the camera's horizontal field of view with --fov if it isn't
about 65 degrees (your position comes from your size and place in the image).

Keys (with a window; click it first): space pause / resume, m mirror, q or Esc quit (the current slot
is saved first). Without a window (--no-window): Ctrl-C stops and saves.
"""
import argparse
import multiprocessing
import queue
import re
import sys
from concurrent.futures import ProcessPoolExecutor
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_pose import PoseCamera, status_bar                                  # noqa: E402

import G1_tele                                                                # noqa: E402,F401
from G1_tele.motion.kinematics import Kinematics                               # noqa: E402
from G1_tele.motion.library import MOTION_DIR                                  # noqa: E402
from G1_tele.motion.recording import (DEFAULTS, FPS, HUMAN_DIR, clean_take, describe, load_raw,  # noqa: E402
                                      missing_landmarks, save_clips, save_raw)

GREEN, RED, AMBER, GREY = (80, 220, 80), (60, 60, 235), (60, 200, 255), (200, 200, 200)
LEVEL_POOL = 600                         # frames kept to estimate "upright" (camera tilt) for the session


def settings_from(a):
    return {k: getattr(a, k) for k in DEFAULTS if getattr(a, k, None) is not None}


def next_slot_number(folder):
    nums = [int(m.group(1)) for p in Path(folder).glob("slot_*.npz") if (m := re.match(r"slot_(\d+)", p.stem))]
    return max(nums, default=0) + 1


# ------------------------------------------------------------------ cleaning + saving, in the background
_KIN = {}


def raw_take(slot, name, parts):
    return {**{k: np.array(slot[k]) for k in ("times", "world", "visibility", "image")}, "aspect": slot["aspect"],
            "level_from": slot["level_from"], "name": name, "parts": parts}


def clean_and_save(folder, name, slot, parts, kind, keep_raw, settings, walk=False, hfov=65.0, robot="g1"):
    """Runs in a separate process (inverse kinematics is heavy; a thread would slow the camera)."""
    if robot not in _KIN:
        _KIN[robot] = Kinematics(robot)
    folder = Path(folder)
    times, world, vis, image = (np.array(slot[k]) for k in ("times", "world", "visibility", "image"))
    clips, rep = clean_take(times, world, vis, parts=parts, name=name, level_from=slot["level_from"],
                            kin=_KIN[robot], image=image, aspect=slot["aspect"], walk=walk, hfov=hfov, **settings)
    folder.mkdir(parents=True, exist_ok=True)
    if clips:
        save_clips(folder / f"{name}.npz", clips, kind=kind, robot=robot)
    if keep_raw:
        save_raw(folder / f"{name}_raw.npz", [raw_take(slot, name, parts)])
    return rep, bool(clips)


class SlotWriter(threading.Thread):
    """Hands finished slots, one at a time, to a cleaning process and keeps the totals."""

    def __init__(self, folder, a):
        super().__init__(daemon=True)
        self.folder, self.a = Path(folder), a
        self.queue, self.lock = queue.Queue(), threading.Lock()
        self.number = next_slot_number(self.folder)
        self.saved_s, self.saved_slots, self.empty_slots, self.last = 0.0, 0, 0, ""
        self.pool = self._new_pool()

    @staticmethod
    def _new_pool():
        pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        pool.submit(int)                                    # start the worker now, not at the first slot
        return pool

    def _clean(self, name, slot):
        args = (str(self.folder), name, slot, self.a.parts, self.a.kind, not self.a.no_raw, settings_from(self.a),
                self.a.walk, self.a.fov, self.a.robot)
        try:
            return self.pool.submit(clean_and_save, *args).result()
        except Exception:                                   # noqa: BLE001  the worker died: one fresh try
            self.pool.shutdown(wait=False)
            self.pool = self._new_pool()
            return self.pool.submit(clean_and_save, *args).result()

    def run(self):
        while True:
            slot = self.queue.get()
            if slot is None:
                break
            name = f"slot_{self.number:04d}"
            self.number += 1
            try:
                rep, kept = self._clean(name, slot)
                msg = f"{name}: {describe(rep)}"
                with self.lock:
                    self.saved_s += rep["kept_s"]
                    self.saved_slots += kept
                    self.empty_slots += not kept
                    self.last = msg
            except Exception as e:                                         # noqa: BLE001
                msg = f"{name} could not be cleaned ({e})"
                try:                                        # keep the raw points so --reprocess can recover it
                    self.folder.mkdir(parents=True, exist_ok=True)
                    save_raw(self.folder / f"{name}_raw.npz", [raw_take(slot, name, self.a.parts)])
                    msg += f"; raw points kept in {name}_raw.npz for --reprocess"
                except Exception:                           # noqa: BLE001
                    pass
                with self.lock:
                    self.last = msg
            print(msg, flush=True)

    def stop(self):
        self.queue.put(None)
        self.join()
        self.pool.shutdown()


# ------------------------------------------------------------------ record
def record(a):
    folder = Path(a.out_dir) / a.name
    writer = SlotWriter(folder, a)
    writer.start()
    if writer.number > 1:
        print(f"{folder} already has slots up to slot_{writer.number - 1:04d}: new slots are added")
    camera = PoseCamera(a.video if a.video else a.camera, loop=False, complexity=a.complexity,
                        width=a.cam_width, height=a.cam_height)
    camera.start()

    slot, last_good, level_pool, n_good = None, None, [], 0
    noise_slots, paused, mirror = 0, False, not a.no_mirror and not a.video
    last_n, t_start, t_print = 0, time.perf_counter(), 0.0
    frame_msg, frame_col = "waiting for the camera...", GREY

    reasons = {"lost": "lost the body points", "long": f"reached --max-slot {a.max_slot:.0f} s",
               "paused": "paused", "end": "recording stopped", "gap": "lost the body points"}

    def close_slot(reason):
        nonlocal slot, noise_slots
        if slot is None:
            return
        dur = slot["times"][-1] - slot["times"][0] if len(slot["times"]) > 1 else 0.0
        clock = f"[{(slot['times'][0] - t_start) / 60:5.1f} min]"
        if dur >= a.min_seconds:
            slot["level_from"] = np.array(level_pool) if level_pool else None
            writer.queue.put(slot)
            print(f"{clock} slot of {dur:.1f} s ended ({reasons[reason]}): saving", flush=True)
        else:
            noise_slots += 1
            print(f"{clock} {dur:.1f} s slot thrown away (under {a.min_seconds:.1f} s, {reasons[reason]})", flush=True)
        slot = None

    window = not a.no_window
    print("recording automatically whenever every needed body point is visible" +
          ("   (space pause, m mirror, q quit)" if window else "   (Ctrl-C to stop)"))
    try:
        while True:
            if camera.error:
                raise SystemExit(camera.error)
            key = cv2.waitKey(1) & 0xFF if window else -1
            now = time.perf_counter()
            det = camera.latest
            if det is not None and det["n"] != last_n:                    # ---- a new camera frame
                last_n = det["n"]
                now, world, vis, image = det["t"], det["world"], det["vis"], det.get("image")
                missing = None if world is None else missing_landmarks(vis, a.parts, a.min_visibility)
                good = world is not None and not missing and not paused
                if paused:
                    frame_msg, frame_col = "paused", AMBER
                elif world is None:
                    frame_msg, frame_col = "nobody found", RED
                elif missing:
                    frame_msg, frame_col = "missing: " + ", ".join(m.replace("_", " ") for m in missing), RED
                else:
                    frame_msg, frame_col = "all body points visible", GREEN
                if good:
                    if slot is not None and now - last_good > a.max_gap_slot:
                        close_slot("gap")
                    if slot is None:
                        slot = {"times": [], "world": [], "visibility": [], "image": [], "aspect": det.get("aspect")}
                    slot["times"].append(now); slot["world"].append(world); slot["visibility"].append(vis)
                    slot["image"].append(image if image is not None else np.full((33, 2), np.nan))
                    last_good = now
                    n_good += 1
                    if n_good % 5 == 0:                                   # posture samples for levelling
                        level_pool.append(world)
                        del level_pool[:-LEVEL_POOL]
                    if slot["times"][-1] - slot["times"][0] >= a.max_slot:
                        close_slot("long")                                # saved in pieces
                if window:
                    draw(a, det, slot, writer, noise_slots, frame_msg, frame_col, mirror, paused)
            else:
                time.sleep(0.003)

            # ---- lost the person for longer than max_gap: the slot ends
            if slot is not None and now - last_good > a.max_gap_slot:
                close_slot("lost")
            if not window and now - t_print > 10:
                t_print = now
                cur = f"slot {slot['times'][-1] - slot['times'][0]:.1f} s" if slot and len(slot["times"]) > 1 else "no slot"
                print(f"[{(now - t_start) / 60:5.1f} min] {frame_msg:40s} {cur:14s} saved {writer.saved_s / 60:.1f} min "
                      f"in {writer.saved_slots} slots, {noise_slots} too short", flush=True)
            if key == ord(" "):
                paused = not paused
                if paused:
                    close_slot("paused")
            elif key == ord("m"):
                mirror = not mirror
            elif key in (ord("q"), 27):
                break
            if camera.ended:
                break
            if a.stop_after and now - t_start > 60 * a.stop_after:
                print(f"--stop-after {a.stop_after} min reached")
                break
    except KeyboardInterrupt:
        print("stopping")
    finally:
        close_slot("end")
        camera.stop()
        if window:
            cv2.destroyAllWindows()
        if not writer.queue.empty():
            print("saving the last slot(s)...")
        writer.stop()
    print(f"{writer.saved_s / 60:.1f} min of motion saved in {writer.saved_slots} slot(s) in {folder}"
          f" ({noise_slots} slot(s) under {a.min_seconds:.1f} s thrown away"
          f"{f', {writer.empty_slots} rejected while cleaning' if writer.empty_slots else ''})")


def draw(a, det, slot, writer, noise_slots, frame_msg, frame_col, mirror, paused):
    cam = det["frame"]
    cam = cv2.resize(cam, (int(cam.shape[1] * a.size / cam.shape[0]), a.size))
    if mirror:
        cam = np.ascontiguousarray(cam[:, ::-1])
    if paused:
        head = ("PAUSED   space: resume   q: quit", AMBER)
    elif slot is not None:
        dur = slot["times"][-1] - slot["times"][0] if len(slot["times"]) > 1 else 0.0
        head = (f"RECORDING  slot {dur:5.1f} s" + ("   (keeps it from 2 s)" if dur < a.min_seconds else "") +
                "   space: pause   q: quit", RED)
    else:
        head = ("WATCHING   records when every needed body point is visible   space: pause   q: quit", GREY)
    with writer.lock:
        saved, slots, last = writer.saved_s, writer.saved_slots, writer.last
    waiting = writer.queue.qsize()
    lines = [head, (frame_msg, frame_col),
             (f"saved {saved / 60:.1f} min in {slots} slot(s), {noise_slots} too short"
              + (f", {waiting} being cleaned" if waiting else ""), GREY)]
    if last:
        lines.append((last[:140], GREY))
    img = np.vstack([status_bar(lines, cam.shape[1]), cam])
    if slot is not None and not paused:
        cv2.circle(img, (img.shape[1] - 20, 20), 8, RED, -1)
    cv2.imshow("record motion", img)


# ------------------------------------------------------------------ clean saved slots again
def reprocess(a):
    target = Path(a.reprocess)
    raws = sorted(target.rglob("*_raw.npz")) if target.is_dir() else [target]
    if not raws:
        raise SystemExit(f"no *_raw.npz files in {target}")
    kin, total, kept_files = Kinematics(a.robot), 0.0, 0
    for raw in raws:
        out = raw.with_name(raw.name.replace("_raw.npz", ".npz"))
        clips = []
        for tk in load_raw(raw):
            if a.walk and tk.get("image") is None:
                print(f"{tk['name']}: recorded before image landmarks were saved - its walking can't be recovered, "
                      "cleaned as poses only")
            c, rep = clean_take(tk["times"], tk["world"], tk["visibility"], parts=a.parts or tk["parts"],
                                name=tk["name"], level_from=tk["level_from"], kin=kin, image=tk.get("image"),
                                aspect=tk.get("aspect"), walk=a.walk and tk.get("image") is not None, hfov=a.fov,
                                **settings_from(a))
            print(f"{tk['name']}: {describe(rep)}")
            clips += c
        if clips:
            save_clips(out, clips, kind=a.kind, robot=a.robot)
            total += sum(len(c["joints"]) for c in clips) / FPS
            kept_files += 1
        elif out.exists():
            out.unlink()                                  # nothing passes the new settings
    print(f"{total / 60:.1f} min kept in {kept_files} file(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="folder name for this recording (under --out-dir)")
    ap.add_argument("--robot", default="g1", help="robot to retarget the motions onto (default g1)")
    ap.add_argument("--out-dir", default=None,
                    help="where the folders go (default motions/human, or motions/<robot>/human for other robots)")
    ap.add_argument("--parts", default=None, choices=("full", "upper"),
                    help="full body (default), or upper body only with the legs kept standing")
    ap.add_argument("--kind", default=None,
                    help='clip kind stored in the files (used to weight training; default "human", "human_walk" with --walk)')
    ap.add_argument("--walk", action="store_true", help="keep where you walk and turn too (for walking policies)")
    ap.add_argument("--fov", type=float, default=65.0, help="the camera's horizontal field of view in degrees (--walk)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--video", default=None, help="a video file instead of the webcam")
    ap.add_argument("--cam-width", type=int, default=None)
    ap.add_argument("--cam-height", type=int, default=None)
    ap.add_argument("--complexity", type=int, default=2, choices=(0, 1, 2),
                    help="MediaPipe model; 2 (most accurate) is the default for recording")
    ap.add_argument("--max-slot", type=float, default=60.0, help="save long slots in pieces of this many seconds")
    ap.add_argument("--stop-after", type=float, default=None, help="stop after this many minutes")
    ap.add_argument("--no-window", action="store_true", help="no display (e.g. over SSH); Ctrl-C stops")
    ap.add_argument("--no-raw", action="store_true", help="don't keep the raw body points (no --reprocess later)")
    ap.add_argument("--size", type=int, default=540, help="window height in pixels")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--reprocess", default=None, help="a folder (or one _raw.npz) to clean again with the settings below")
    g = ap.add_argument_group("cleaning (defaults in G1_tele/motion/recording.py)")
    g.add_argument("--min-visibility", type=float, default=DEFAULTS["min_visibility"],
                   help="how sure MediaPipe must be of a body point (0-1)")
    g.add_argument("--min-seconds", type=float, default=DEFAULTS["min_seconds"],
                   help="shortest slot / clip kept (s); shorter ones are noise")
    g.add_argument("--max-gap", type=float, default=DEFAULTS["max_gap"],
                   help="how long the body points may be lost before a slot ends (s); shorter holes are filled in")
    g.add_argument("--max-spike", type=float, default=None, help="glitch threshold (m)")
    g.add_argument("--max-speed", type=float, default=None, help="fastest keypoint movement kept (m/s)")
    g.add_argument("--smooth", type=float, default=None, help="smoothing (s, Gaussian sigma)")
    g.add_argument("--max-ik-err", type=float, default=None, help="largest IK error kept (m)")
    g.add_argument("--max-joint-speed", type=float, default=None, help="fastest joint movement kept (rad/s)")
    a = ap.parse_args()
    from G1_tele.robots import ROBOTS
    if a.robot not in ROBOTS:
        ap.error(f"unknown robot {a.robot}; known: {sorted(ROBOTS)}")
    if a.out_dir is None:
        a.out_dir = str(HUMAN_DIR if a.robot == "g1" else MOTION_DIR / a.robot / "human")
    a.kind = a.kind or ("human_walk" if a.walk else "human")
    a.max_gap_slot = a.max_gap
    if a.reprocess:
        return reprocess(a)
    if not a.name:
        ap.error("--name is required (or use --reprocess)")
    a.parts = a.parts or "full"
    record(a)


if __name__ == "__main__":
    main()
