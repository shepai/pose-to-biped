"""
src/G1_tele/motion/recording.py  -  turning camera recordings into clean training motions

A take is what MediaPipe gave for one recording: frame times (T,), pose_world_landmarks (T, 33, 3)
and visibility (T, 33). clean_take() keeps only the parts that can be trusted and turns them into
motion-library clips (root + 29 joints at 50 Hz, the same format as synthetic.npz):

  1. Frame check: every landmark the chosen body parts need must be visible enough. parts="full"
     needs shoulders, elbows, wrists, hips, knees and ankles; parts="upper" leaves out the legs,
     which are then put in the robot's standing pose. Other frames are dropped.
  2. Glitch check: MediaPipe sometimes throws a point away for a frame or two. A frame where a
     landmark sits off the straight line between the frames either side of it by more than
     max_spike (m), or by more than twice this take's typical amount if that is larger, is dropped;
     so is a frame whose robot keypoints moved faster than max_speed (m/s) since the last kept
     frame. (Real movement filmed at 30 fps stays within a few cm of that line, even when a hand
     changes direction fast.) The direction the person faces is smoothed over heading_smooth
     seconds, because the hips are close together and their jitter would otherwise turn the pose.
  3. Segments: kept frames less than max_gap seconds apart form one segment, and the small holes
     between them are filled in. A longer hole ends the segment: a clip never spans a gap, so it
     never jumps from one pose to another.
  4. Each segment is resampled to 50 Hz and smoothed without delay (a Gaussian filter applied over
     the whole segment), then turned into robot poses with inverse kinematics.
  5. Pose check: frames the robot can't reach (IK error above max_ik_err, m) or where a joint
     would move faster than max_joint_speed (rad/s) split the segment again.
  6. Segments shorter than min_seconds are dropped.

walk=True (needs the image landmarks, (T, 33, 2), and the image's width / height): the clips also
carry where the person walked and which way they faced (HumanToRobot.room_pose), relative to the
start of each clip, for training walking policies (train_locomotion.py). Frames whose position
can't be worked out are dropped, a position moving faster than max_root_speed (m/s) or a heading
turning faster than max_turn_rate (rad/s) counts as a glitch, and the path is smoothed over
root_smooth seconds (depth from one camera is noisy). Face the camera roughly (up to side-on): with
their back to it, MediaPipe often swaps a person's left and right.

The report says how much was kept and why the rest was dropped.
"""
from pathlib import Path

import numpy as np

from G1_tele.motion.human import LM, HumanToRobot, place_in_room
from G1_tele.motion.kinematics import ANKLE_HEIGHT, KEYPOINT_SETS, G1Kinematics
from G1_tele.motion.library import MOTION_DIR, MotionLibrary, save_library
from G1_tele.motion.retarget import KeypointRetargeter

FPS = 50.0                                   # the motion library rate (synthetic.npz uses 50 Hz too)
HUMAN_DIR = MOTION_DIR / "human"
NAMES = list(KEYPOINT_SETS["full"])
_ARMS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
_LEGS = ["left_knee", "right_knee", "left_ankle", "right_ankle"]
REQUIRED = {"full": _ARMS + ["left_hip", "right_hip"] + _LEGS, "upper": _ARMS + ["left_hip", "right_hip"]}
_LEG_LANDMARKS = [LM[n] for n in _LEGS]

DEFAULTS = dict(min_visibility=0.5, max_spike=0.08, heading_smooth=0.15, max_speed=6.0, max_gap=0.15, smooth=0.04, max_ik_err=0.06,
                max_joint_speed=12.0, min_seconds=2.0, max_root_speed=3.0, max_turn_rate=6.0, root_smooth=0.12)


def missing_landmarks(visibility, parts="full", min_visibility=0.5):
    """Names of the landmarks this frame lacks (hips need only half the visibility: a desk often
    hides them, and MediaPipe still places them well)."""
    v = np.asarray(visibility, float)
    return [n for n in REQUIRED[parts]
            if v[LM[n]] < (0.5 * min_visibility if n.endswith("hip") else min_visibility)]


def _gaussian_smooth(x, sigma):
    """Zero-phase Gaussian smoothing along axis 0 (sigma in samples), edges reflected."""
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([x[r:0:-1], x, x[-2:-r - 2:-1]]) if len(x) > r + 1 else np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    flat = pad.reshape(len(pad), -1)
    out = np.stack([np.convolve(flat[:, i], k, mode="valid") for i in range(flat.shape[1])], axis=1)
    return out.reshape((len(x),) + x.shape[1:])


def _split(mask):
    """Runs of True in a boolean array -> list of (start, stop) index pairs."""
    runs, start = [], None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def clean_take(times, world, visibility, parts="full", name="take", level_from=None, kin=None,
               com_weight=0.0, image=None, aspect=None, walk=False, hfov=65.0, **settings):
    """Returns (clips, report). clips: list of dicts with root (T,7), joints (T,29), name, ik_err.
    level_from: landmarks (N, 33, 3) of the person standing upright (e.g. during a countdown), used
    to level the camera; if None, the first half second of good frames is used.
    walk: also keep where the person walked (needs image (T, 33, 2) and aspect = width / height)."""
    s = {**DEFAULTS, **settings}
    if walk and (image is None or not aspect):
        raise ValueError("walk=True needs the image landmarks and the image's aspect ratio "
                         "(recorded with record_motion.py from this version on)")
    times, world, vis = np.asarray(times, float), np.asarray(world, float), np.asarray(visibility, float)
    kin = kin or G1Kinematics()
    conv = HumanToRobot(kin, s["min_visibility"])
    T = len(times)
    report = {"seconds": float(times[-1] - times[0]) if T > 1 else 0.0, "frames": T, "missing_s": 0.0, "absent_s": 0.0,
              "jump_frames": 0, "unreachable_frames": 0, "too_fast_frames": 0, "short_s": 0.0, "clips": []}
    if T < 2:
        return [], report
    frame_dt = np.diff(times, append=times[-1] + np.median(np.diff(times)))
    nominal = float(np.median(np.diff(times)))
    report["absent_s"] = float(np.sum(np.clip(np.diff(times) - nominal, 0, None)[np.diff(times) > s["max_gap"]]))

    # 1. frame check
    ok = np.array([not missing_landmarks(v, parts, s["min_visibility"]) for v in vis])
    report["missing_s"] = float(frame_dt[~ok].sum())
    if parts == "upper":                                    # legs -> standing pose
        vis = vis.copy()
        vis[:, _LEG_LANDMARKS] = 0.0
    good = np.flatnonzero(ok)
    if len(good) == 0:
        return [], report
    if level_from is not None and len(level_from):
        conv.calibrate(np.asarray(level_from, float))
    else:
        first = good[times[good] <= times[good[0]] + 0.5]
        conv.calibrate(world[first])

    # 2. glitches: landmarks off the line between their neighbours
    used = sorted({LM[n] for n in REQUIRED[parts]} | {LM[n] for n in ("left_index", "right_index", "left_pinky", "right_pinky")})
    P, tg = world[good][:, used], times[good]
    resid = np.zeros(len(good))
    for k in range(1, len(good) - 1):
        ta, tb = tg[k - 1], tg[k + 1]
        if tb - ta <= 2 * s["max_gap"]:
            f = (tg[k] - ta) / (tb - ta)
            resid[k] = np.max(np.linalg.norm(P[k] - ((1 - f) * P[k - 1] + f * P[k + 1]), axis=1))
    typical = float(np.median(resid[1:-1])) if len(good) > 2 else 0.0
    spike = resid > max(s["max_spike"], 2.0 * typical)
    report["jump_frames"] += int(spike.sum())
    good, tg = good[~spike], tg[~spike]

    # facing direction from the hips, smoothed over time (unwrapped angle, Gaussian in time)
    L = np.array([conv.levelled(world[i])[LM["left_hip"]] - conv.levelled(world[i])[LM["right_hip"]] for i in good])
    yaw = np.unwrap(np.arctan2(L[:, 1], L[:, 0]))
    if s["heading_smooth"] > 0 and len(good) > 2:
        sig = s["heading_smooth"] / max(float(np.median(np.diff(tg))), 1e-3)
        yaw = _gaussian_smooth(yaw[:, None], sig)[:, 0]
    lefts = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], 1)

    # robot keypoints for every remaining frame
    kps, prev, room = [], None, []
    for k, i in enumerate(good):
        if prev is not None and times[i] - times[prev] > s["max_gap"]:
            conv.reset_contacts()
        kp = conv(world[i], vis[i], left=lefts[k])
        kps.append(np.array([kp[n] for n in NAMES])); prev = i
        if walk:
            rp = conv.room_pose(world[i], image[i], vis[i], aspect, hfov)
            room.append([np.nan] * 3 if rp is None else [rp[0][0], rp[0][1], conv.heading])
    kps = np.array(kps)
    if walk:                                               # no position: drop the frame
        room = np.array(room)
        room[:, 2] = np.unwrap(np.nan_to_num(room[:, 2]))
        placed = np.isfinite(room[:, 0])
        report["jump_frames"] += int((~placed).sum())
        good, kps, room = good[placed], kps[placed], room[placed]

    # 3. segments (speed check against the last kept frame)
    segments, seg, last, rejected_in_row = [], [], None, 0
    for k, i in enumerate(good):
        if last is not None and times[i] - times[last] > s["max_gap"]:
            segments.append(seg); seg, last, rejected_in_row = [], None, 0
        x = kps[k]
        if last is not None:
            gap = max(times[i] - times[last], 1e-3)
            speed = np.max(np.linalg.norm(x - seg[-1][1], axis=1)) / gap
            if walk and (np.linalg.norm(room[k, :2] - seg[-1][2][:2]) / gap > s["max_root_speed"]
                         or abs(room[k, 2] - seg[-1][2][2]) / gap > s["max_turn_rate"]):
                speed = np.inf                              # the position or heading jumped
            if speed > s["max_speed"]:
                report["jump_frames"] += 1
                rejected_in_row += 1
                if rejected_in_row >= 3:                    # the kept frame was the odd one out: restart here
                    segments.append(seg[:-1]); seg, last, rejected_in_row = [(times[i], x, room[k] if walk else None)], i, 0
                continue
        rejected_in_row = 0
        seg.append((times[i], x, room[k] if walk else None)); last = i
    segments.append(seg)

    # 4.-6. resample, smooth, IK, pose check
    clips = []
    for seg in segments:
        if len(seg) < 2 or seg[-1][0] - seg[0][0] < s["min_seconds"]:
            report["short_s"] += float(seg[-1][0] - seg[0][0]) if len(seg) > 1 else 0.0
            continue
        ts = np.array([t for t, _, _ in seg])
        xs = np.array([x for _, x, _ in seg])                           # (n, 12, 3)
        grid = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / FPS)
        X = np.stack([[np.interp(grid, ts, xs[:, k, d]) for d in range(3)] for k in range(len(NAMES))], 0)
        X = np.transpose(X, (2, 0, 1))                                  # (m, 12, 3)
        X = _gaussian_smooth(X, s["smooth"] * FPS)
        for f in (NAMES.index("left_foot"), NAMES.index("right_foot")):  # keep planted feet on the floor
            z = X[:, f, 2]
            X[:, f, 2] = np.where(z < ANKLE_HEIGHT + 0.01, ANKLE_HEIGHT, z)
        ik = KeypointRetargeter(kin, com_weight=com_weight)
        roots, joints, errs = [], [], []
        for frame in X:
            r, j, e = ik.step(dict(zip(NAMES, frame)))
            roots.append(r); joints.append(j); errs.append(e)
        roots, joints, errs = np.array(roots), np.array(joints), np.array(errs)
        if walk:                                            # where they walked, relative to the clip start
            rs = np.array([r for _, _, r in seg])
            Rm = np.stack([np.interp(grid, ts, rs[:, d]) for d in range(3)], 1)
            Rm = _gaussian_smooth(Rm, s["root_smooth"] * FPS)
            c0, s0 = np.cos(Rm[0, 2]), np.sin(Rm[0, 2])
            d = Rm[:, :2] - Rm[0, :2]
            rel = np.stack([c0 * d[:, 0] + s0 * d[:, 1], -s0 * d[:, 0] + c0 * d[:, 1], Rm[:, 2] - Rm[0, 2]], 1)
            roots = np.array([place_in_room(r, p) for r, p in zip(roots, rel)])
        reach = errs <= s["max_ik_err"]
        speed = np.r_[0.0, np.max(np.abs(np.diff(joints, axis=0)), axis=1) * FPS]
        calm = speed <= s["max_joint_speed"]
        report["unreachable_frames"] += int((~reach).sum())
        report["too_fast_frames"] += int((reach & ~calm).sum())
        for a, b in _split(reach & calm):
            dur = (b - a - 1) / FPS
            if dur < s["min_seconds"]:
                report["short_s"] += max(dur, 0.0)
                continue
            clip = {"root": roots[a:b], "joints": joints[a:b], "ik_err": float(errs[a:b].mean()),
                    "name": f"{name}_{len(clips)}", "t0": float(grid[a])}
            clips.append(clip)
            report["clips"].append({"name": clip["name"], "seconds": round(dur, 2),
                                    "ik_err_cm": round(100 * clip["ik_err"], 1)})
    report["kept_s"] = round(sum(c["seconds"] for c in report["clips"]), 2)
    return clips, report


def describe(report):
    """One-paragraph summary of a clean_take() report."""
    lines = [f"{report['seconds']:.1f} s recorded, {report['kept_s']:.1f} s kept as {len(report['clips'])} clip(s)"]
    why = []
    if report.get("absent_s", 0) > 0.05:
        why.append(f"{report['absent_s']:.1f} s with nobody found")
    if report["missing_s"] > 0.05:
        why.append(f"{report['missing_s']:.1f} s with body points missing")
    if report["jump_frames"]:
        why.append(f"{report['jump_frames']} glitch frame(s)")
    if report["unreachable_frames"]:
        why.append(f"{report['unreachable_frames']} frame(s) the robot can't reach")
    if report["too_fast_frames"]:
        why.append(f"{report['too_fast_frames']} frame(s) moving too fast")
    if report["short_s"] > 0.05:
        why.append(f"{report['short_s']:.1f} s in pieces shorter than the minimum")
    if why:
        lines.append("dropped: " + ", ".join(why))
    return "; ".join(lines)


# ------------------------------------------------------------------ files
def save_clips(path, clips, kind="human", keep_existing=None):
    """Write clips as a motion library (50 Hz). keep_existing: clips already in the file to keep,
    as returned by load_clips()."""
    allc = list(keep_existing or []) + list(clips)
    save_library(path, [(c["root"], c["joints"]) for c in allc], FPS,
                 names=[c["name"] for c in allc], kinds=[c.get("kind", kind) for c in allc])


def load_clips(path):
    """Clips of an existing motion library as dicts (for appending to it)."""
    if not Path(path).exists():
        return []
    lib = MotionLibrary(path)
    return [{"root": lib.root[lib.starts[c]:lib.starts[c + 1]], "joints": lib.joints[lib.starts[c]:lib.starts[c + 1]],
             "name": str(lib.names[c]), "kind": str(lib.kinds[c])} for c in range(lib.n_clips)]


def save_raw(path, takes):
    """Raw takes (so they can be cleaned again with other settings): list of dicts with
    times, world, visibility, level_from, name, parts."""
    data = {}
    for i, tk in enumerate(takes):
        for key in ("times", "world", "visibility"):
            data[f"{key}_{i}"] = np.asarray(tk[key], np.float32 if key != "times" else np.float64)
        if tk.get("image") is not None:                     # image landmarks: needed to recover walking
            data[f"image_{i}"] = np.asarray(tk["image"], np.float32)
            data[f"aspect_{i}"] = np.float32(tk["aspect"])
        data[f"level_{i}"] = np.asarray(tk.get("level_from") if tk.get("level_from") is not None else np.zeros((0, 33, 3)), np.float32)
        data[f"name_{i}"] = np.array(tk["name"])
        data[f"parts_{i}"] = np.array(tk["parts"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n_takes=len(takes), **data)


def load_raw(path):
    f = np.load(path, allow_pickle=False)
    return [{"times": f[f"times_{i}"], "world": f[f"world_{i}"], "visibility": f[f"visibility_{i}"],
             "level_from": f[f"level_{i}"] if len(f[f"level_{i}"]) else None,
             "name": str(f[f"name_{i}"]), "parts": str(f[f"parts_{i}"]),
             "image": f[f"image_{i}"] if f"image_{i}" in f else None,
             "aspect": float(f[f"aspect_{i}"]) if f"aspect_{i}" in f else None} for i in range(int(f["n_takes"]))]
