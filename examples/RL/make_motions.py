"""
Build motion libraries for the pose-tracking env.

    # synthetic, balanced whole-body motions (default output: G1_tele/assets/motions/synthetic.npz)
    python make_motions.py synth --clips-per-kind 30 --seconds 20

    # your old hands.npy (T,2,3 hand-relative-to-shoulder) -> full-body reference clips
    python make_motions.py hands /path/to/hands.npy --fps 30 --out my_hands.npz

    # keypoint clips from a tracker: .npz with one (T,3) array per keypoint name
    # (left_hand, right_hand, left_foot, right_foot, optional elbows/knees/hips/shoulders),
    # already in the robot heading frame and robot scale
    python make_motions.py keypoints /path/to/clip.npz --fps 30 --out my_clip.npz

    # combine several libraries into one
    python make_motions.py merge out.npz a.npz b.npz
"""
import argparse
import time

import numpy as np

from G1_tele.motion.kinematics import G1Kinematics
from G1_tele.motion.library import DEFAULT_MOTIONS, merge_libraries, save_library
from G1_tele.motion.retarget import hands_relative_to_shoulders, resample, retarget_clip
from G1_tele.motion.synthetic import SyntheticMotionGenerator

FPS = 50   # library frame rate (= env control rate)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", default=str(DEFAULT_MOTIONS))
    s.add_argument("--kinds", default="upper,whole,balance,step")
    s.add_argument("--clips-per-kind", type=int, default=30)
    s.add_argument("--seconds", type=float, default=20.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--xml", default=None)
    for name in ("hands", "keypoints"):
        h = sub.add_parser(name)
        h.add_argument("path")
        h.add_argument("--fps", type=float, required=True, help="frame rate of the input data")
        h.add_argument("--out", required=True)
        h.add_argument("--xml", default=None)
    mg = sub.add_parser("merge")
    mg.add_argument("out")
    mg.add_argument("inputs", nargs="+")
    a = p.parse_args()

    if a.cmd == "merge":
        merge_libraries(a.out, *a.inputs)
        print("wrote", a.out)
        return
    kin = G1Kinematics(a.xml) if a.xml else G1Kinematics()
    t0 = time.time()
    if a.cmd == "synth":
        gen = SyntheticMotionGenerator(kin, fps=FPS, seed=a.seed)
        clips, names, kinds = gen.library(a.kinds.split(","), a.clips_per_kind, a.seconds)
        save_library(a.out, clips, FPS, names, kinds)
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
        save_library(a.out, clips, FPS, names, ["upper" if a.cmd == "hands" else "keypoints"] * len(clips))
    print(f"wrote {a.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
