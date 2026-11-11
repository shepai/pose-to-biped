"""
Build motion libraries for the pose-tracking env.

    # synthetic, balanced whole-body motions (default output: G1_tele/assets/motions/synthetic.npz)
    python make_motions.py synth --clips-per-kind 30 --seconds 20

    # walking and turning (kinds turn, walk, loco; default output: G1_tele/assets/motions/locomotion.npz)
    python make_motions.py loco --clips-per-kind 30 --seconds 20

    # your old hands.npy (T,2,3 hand-relative-to-shoulder) -> full-body reference clips
    python make_motions.py hands /path/to/hands.npy --fps 30 --out my_hands.npz

    # keypoint clips from a tracker: .npz with one (T,3) array per keypoint name
    # (left_hand, right_hand, left_foot, right_foot, optional elbows/knees/hips/shoulders),
    # already in the robot heading frame and robot scale
    python make_motions.py keypoints /path/to/clip.npz --fps 30 --out my_clip.npz

    # combine several libraries into one
    python make_motions.py merge out.npz a.npz b.npz

Other robots: add --robot h1 (t1, n1, ...) to any command. Default outputs are the robot's own files,
G1_tele/assets/motions/<robot>/synthetic.npz and locomotion.npz, which the env and the training
scripts pick up by themselves. Every file records which robot it is for.

    python make_motions.py synth --robot h1 --clips-per-kind 30
    python make_motions.py loco --robot h1 --clips-per-kind 30
    python make_motions.py synth --robot all          # every robot in G1_tele.robots, one after the other
"""
import argparse
import time

import numpy as np

from G1_tele.motion.kinematics import Kinematics
from G1_tele.motion.gait import LOCO_KINDS, GaitGenerator
from G1_tele.motion.library import default_motions, merge_libraries, save_library
from G1_tele.robots import ROBOTS
from G1_tele.motion.retarget import hands_relative_to_shoulders, resample, retarget_clip
from G1_tele.motion.synthetic import SyntheticMotionGenerator

FPS = 50   # library frame rate (= env control rate)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", default=None, help="default: the robot's synthetic.npz")
    s.add_argument("--kinds", default="upper,whole,balance,step")
    s.add_argument("--clips-per-kind", type=int, default=30)
    s.add_argument("--seconds", type=float, default=20.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--xml", default=None)
    lo = sub.add_parser("loco")
    lo.add_argument("--out", default=None, help="default: the robot's locomotion.npz")
    lo.add_argument("--kinds", default=",".join(LOCO_KINDS))
    lo.add_argument("--clips-per-kind", type=int, default=30)
    lo.add_argument("--seconds", type=float, default=20.0)
    lo.add_argument("--seed", type=int, default=0)
    lo.add_argument("--xml", default=None)
    for name in ("hands", "keypoints"):
        h = sub.add_parser(name)
        h.add_argument("path")
        h.add_argument("--fps", type=float, required=True, help="frame rate of the input data")
        h.add_argument("--out", required=True)
        h.add_argument("--xml", default=None)
    for sp in (s, lo, *[sub.choices[n] for n in ("hands", "keypoints")]):
        sp.add_argument("--robot", default="g1", help=f"one of {sorted(ROBOTS)} (synth/loco: or 'all')")
    mg = sub.add_parser("merge")
    mg.add_argument("out")
    mg.add_argument("inputs", nargs="+")
    a = p.parse_args()

    if a.cmd == "merge":
        merge_libraries(a.out, *a.inputs)
        print("wrote", a.out)
        return
    robots = sorted(ROBOTS) if a.robot == "all" else [a.robot]
    if len(robots) > 1 and (a.cmd not in ("synth", "loco") or a.out or a.xml):
        raise SystemExit("--robot all works with synth / loco and their default outputs only")
    for robot in robots:
        make(a, robot)


def make(a, robot):
    kin = Kinematics(robot, a.xml)
    print(f"[{kin.spec.name}] {kin.n_joints} joints, scale {kin.scale:.2f} x the G1", flush=True)
    t0 = time.time()
    if a.cmd in ("synth", "loco"):
        a_out = a.out or str(default_motions(kin.spec.name, loco=a.cmd == "loco"))
    else:
        a_out = a.out
    if a.cmd == "synth":
        gen = SyntheticMotionGenerator(kin, fps=FPS, seed=a.seed)
        clips, names, kinds = gen.library(a.kinds.split(","), a.clips_per_kind, a.seconds)
        save_library(a_out, clips, FPS, names, kinds, robot=kin.spec)
    elif a.cmd == "loco":
        gen = GaitGenerator(kin, fps=FPS, seed=a.seed)
        clips, names, kinds = gen.library(a.kinds.split(","), a.clips_per_kind, a.seconds)
        save_library(a_out, clips, FPS, names, kinds, robot=kin.spec)
        print(f"{gen.stats['clips']} clips from {gen.stats['attempts']} attempts")
    else:
        raw = np.load(a.path, allow_pickle=True)
        if a.cmd == "hands":
            if isinstance(raw, np.lib.npyio.NpzFile):
                seqs = [raw[f] for f in raw.files]
            elif raw.dtype == object:
                seqs = list(raw)
            else:
                seqs = [raw]
            kps = [hands_relative_to_shoulders(s, kin) for s in seqs]
        else:
            kps = [{n: raw[n].astype(float) for n in raw.files}]
        clips, names = [], []
        for i, kp in enumerate(kps):
            kp = resample(kp, a.fps, FPS)
            root, joints, err = retarget_clip(kp, kin)
            print(f"  clip {i}: {len(root) / FPS:.1f} s, retarget error mean {err.mean() * 100:.1f} cm, "
                  f"max {err.max() * 100:.1f} cm")
            clips.append((root, joints)); names.append(f"{a.cmd}_{i:03d}")
        save_library(a_out, clips, FPS, names, ["upper" if a.cmd == "hands" else "keypoints"] * len(clips),
                     robot=kin.spec)
    print(f"wrote {a_out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
