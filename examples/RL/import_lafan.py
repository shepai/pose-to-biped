"""
examples/RL/import_lafan.py  -  real human walking for the G1, from Unitree's retargeted LAFAN1 dataset

    # 1. download the G1 files (CC BY-NC-ND 4.0: fine to train on, don't redistribute them or the
    #    converted files - keep them out of a public repository)
    #    https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset  ->  the g1/ folder
    # 2. convert the walking clips (default: files whose name starts with "walk")
    python examples/RL/import_lafan.py path/to/LAFAN1_Retargeting_Dataset/g1
    python examples/RL/import_lafan.py path/to/g1 --include walk run --speed 0.8     # more, a bit slower
    # 3. train with them (kind "lafan": unlocked after the synthetic walking, at the end of the curriculum)
    python examples/RL/train_locomotion.py --from ... --name ... --motions src/G1_tele/assets/motions/lafan

Each CSV row is one 30 fps frame: root position x, y, z, root quaternion qx, qy, qz, qw, then the 29
joint angles in Unitree's order (the same joints as this repository's G1). The conversion:
  * reorders the quaternion to (w, x, y, z) and the joints by name,
  * plays the motion at --speed (below 1 = slower, easier to follow) and resamples it to 50 Hz,
  * moves each clip up or down so the feet stand on this model's floor,
  * clamps joints to this model's limits,
  * cuts long recordings into pieces of at most --chunk seconds (episodes start at random times in a
    clip, so this keeps every part of a long recording equally likely).
The retargeting is kinematic (no physics, no motor limits), so not every frame is something the
robot can do exactly; the residual policy has to make up the difference, which is why this data sits at
the end of the curriculum.
"""
import argparse
import fnmatch
from pathlib import Path

import numpy as np

import G1_tele  # noqa: F401
from G1_tele.envs.whole_body import BODY_JOINTS
from G1_tele.motion.kinematics import ANKLE_HEIGHT, G1Kinematics, slerp
from G1_tele.motion.library import MOTION_DIR, save_library

LAFAN_FPS = 30.0
FPS = 50.0
LAFAN_G1_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right")
                   for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")] + \
                  ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + \
                  [f"{s}_{j}_joint" for s in ("left", "right")
                   for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch",
                             "wrist_yaw")]
ORDER = [LAFAN_G1_JOINTS.index(n) for n in BODY_JOINTS]


def read_csv(path):
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] != 7 + 29:
        raise ValueError(f"{path}: {data.shape[1]} columns, expected 36 (root 7 + 29 G1 joints)")
    pos = data[:, :3]
    quat = data[:, [6, 3, 4, 5]]                           # (qx, qy, qz, qw) -> (w, x, y, z)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    for i in range(1, len(quat)):                          # no sign flips (for slerp)
        if quat[i] @ quat[i - 1] < 0:
            quat[i] = -quat[i]
    return pos, quat, data[:, 7:][:, ORDER]


def resample(pos, quat, joints, speed):
    t_src = np.arange(len(pos)) / LAFAN_FPS / speed
    t = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / FPS)
    x = np.clip(t * LAFAN_FPS * speed, 0, len(pos) - 1)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, len(pos) - 1)
    f = (x - i0)[:, None]
    P = (1 - f) * pos[i0] + f * pos[i1]
    J = (1 - f) * joints[i0] + f * joints[i1]
    Q = np.array([slerp(quat[a], quat[b], float(ff)) for a, b, ff in zip(i0, i1, f[:, 0])])
    return P, Q, J


def on_floor(kin, root, joints):
    """Shift root height so the lower foot's ankle frame sits ANKLE_HEIGHT above the floor (robust: the
    5th percentile of the per-frame lower-foot height, so the odd frame with a foot through the floor
    doesn't lift the whole clip)."""
    low = []
    for r, j in zip(root[::5], joints[::5]):
        kin.forward(kin.qpos_from(r, j))
        low.append(min(kin.d.xpos[kin.foot_ids, 2]))
    dz = ANKLE_HEIGHT - float(np.percentile(low, 5))
    root = root.copy()
    root[:, 2] += dz
    return root, dz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="the dataset's g1/ folder (or any folder of its CSV files)")
    ap.add_argument("--include", nargs="+", default=["walk"],
                    help="file name prefixes to convert (default: walk); 'all' for every file")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (below 1 = slower)")
    ap.add_argument("--chunk", type=float, default=30.0, help="longest clip in seconds")
    ap.add_argument("--kind", default="lafan", help="clip kind (used by the curriculum and for weighting)")
    ap.add_argument("--out", default=str(MOTION_DIR / "lafan" / "lafan.npz"), help="output motion file")
    a = ap.parse_args()

    files = sorted(Path(a.folder).glob("*.csv"))
    if "all" not in a.include:
        files = [f for f in files if any(fnmatch.fnmatch(f.stem.lower(), p.lower() + "*") for p in a.include)]
    if not files:
        raise SystemExit(f"no matching .csv files in {a.folder} (--include {' '.join(a.include)})")
    kin = G1Kinematics()
    clips, names, total = [], [], 0.0
    for path in files:
        pos, quat, joints = read_csv(path)
        P, Q, J = resample(pos, quat, joints, a.speed)
        J = np.clip(J, kin.q_lo, kin.q_hi)
        root = np.concatenate([P, Q], 1)
        root, dz = on_floor(kin, root, J)
        n = int(round(a.chunk * FPS))
        pieces = [(s, min(s + n, len(root))) for s in range(0, len(root), n)]
        pieces = [(s, e) for s, e in pieces if e - s >= 2 * FPS]        # drop scraps under 2 s
        for k, (s, e) in enumerate(pieces):
            clips.append((root[s:e], J[s:e]))
            names.append(f"{path.stem}_{k}")
        dur = len(root) / FPS
        total += dur
        v = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) * FPS
        jv = np.abs(np.diff(J, axis=0)).max(1) * FPS
        print(f"{path.name:32s} {dur:6.1f} s -> {len(pieces)} clip(s)   speed: median {np.median(v):.2f} m/s, "
              f"top {np.percentile(v, 95):.2f} m/s   fastest joint {np.percentile(jv, 99):.1f} rad/s (99th pct)   "
              f"height shift {100 * dz:+.1f} cm")
    save_library(a.out, clips, FPS, names, [a.kind] * len(clips))
    print(f"wrote {a.out}: {len(clips)} clips, {total / 60:.1f} min at speed {a.speed}")


if __name__ == "__main__":
    main()
