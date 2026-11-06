"""
examples/RL/finetune.py  -  keep training a trained policy on new motions (e.g. your own recordings)

    python examples/RL/finetune.py --from logs/pose_stage1T1/pose_stage1T1 --name pose_human1 \\
        --motions src/G1_tele/assets/motions/human
    python examples/RL/finetune.py --from logs/pose_stage1T1/pose_stage1T1 --name pose_human1 \\
        --motions src/G1_tele/assets/motions/human/arm_waves.npz --steps 20M --new-share 0.8
    python examples/RL/finetune.py ... --robust        # also stage 2's pushes, friction changes and sensor noise
    python examples/RL/finetune.py ... --dry-run       # show the motion mix and test the starting policy on
                                                       # your clips, without training

--from     the trained model (with or without .zip); its _vecnormalize.pkl must sit next to it
--name     the new run: everything goes to logs/<name>/, the final model is logs/<name>/<name>.zip
--motions  .npz motion files, folders of them (searched recursively), or patterns ("motions/human/*.npz"); *_raw.npz are skipped

This is the normal training pipeline (train_pose_tracking.py + run_logging.py): the same environment,
network, PPO settings, logs, evaluation videos and checkpoints. What changes:
  * training starts from --from: its weights and its observation statistics (which keep updating),
  * the motions are your files plus, unless --no-synthetic, the synthetic library, so the policy keeps
    what it already knows. --new-share of the episodes (default 60 %) start on one of your clips;
    the rest are spread evenly over the synthetic kinds (upper, whole, balance, step),
  * no curriculum, and stage 2's smaller learning rate (5e-5 falling to 1e-5; --lr to change it),
  * the evaluation also runs your clips: eval/survival_human in progress.csv (and eval.csv) shows how
    well the policy follows them, next to the synthetic kinds, so you can see if it forgets anything.

Motion files must be 50 Hz like synthetic.npz (record_motion.py and make_motions.py write 50 Hz).

Walking models (from train_locomotion.py) are recognised from their observation size: they are
fine-tuned in the locomotion env, with the walking clips (locomotion.npz) mixed in as well.
"""
import argparse
import glob
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_tracking as T          # noqa: E402  (sets up MuJoCo's OpenGL and registers the G1-pose-* envs)
from run_logging import (GL_INFO, PeriodicCheckpointCallback, PeriodicVideoEvalCallback,  # noqa: E402
                         TrackingStatsCallback, prepare_run_dir, save_config)

import gymnasium as gym                  # noqa: E402
import numpy as np                       # noqa: E402
from stable_baselines3 import PPO        # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from G1_tele.motion.library import DEFAULT_MOTIONS, LOCO_MOTIONS, MotionLibrary  # noqa: E402

MIN_CLIP_SECONDS = 1.5                   # the env only starts episodes on clips at least this long


def parse_steps(text):
    t = str(text).strip().lower().replace("_", "")
    mult = {"k": 1e3, "m": 1e6, "g": 1e9}.get(t[-1:], 1)
    return int(float(t[:-1] if mult != 1 else t) * mult)


def model_base(path):
    base = str(path)[:-4] if str(path).endswith(".zip") else str(path)
    missing = [f for f in (base + ".zip", base + "_vecnormalize.pkl")
               if not os.path.exists(f) and not (f.endswith(".zip") and os.path.exists(base))]
    if missing:
        raise SystemExit("can't find " + " or ".join(os.path.abspath(f) for f in missing))
    return base


def motion_files(items):
    files = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            found = sorted(p.rglob("*.npz"))                 # sub-folders too (record_motion.py writes motions/human/)
        elif any(ch in item for ch in "*?["):
            found = sorted(Path(x) for x in glob.glob(item))
        else:
            if not p.exists():
                raise SystemExit(f"no such motion file: {item}")
            found = [p]
        files += [f.resolve() for f in found if not f.name.endswith("_raw.npz")]
    files = list(dict.fromkeys(files))
    if not files:
        raise SystemExit(f"no motion files found in {' '.join(items)}")
    return files


def is_walking_model(base):
    """True if the model was trained with locomotion=True (train_locomotion.py)."""
    n_obs = int(PPO.load(base, device="cpu").observation_space.shape[0])
    sizes = {}
    for env_id in (T.EVAL_ID, T.LOCO_EVAL):
        kw = {"motion_file": str(DEFAULT_MOTIONS)}
        e = gym.make(env_id, **kw).unwrapped
        sizes[env_id] = int(e.observation_space.shape[0])
        e.close()
    if n_obs == sizes[T.LOCO_EVAL]:
        return True
    if n_obs != sizes[T.EVAL_ID]:
        raise SystemExit(f"{base} expects {n_obs} observations, but the pose env has {sizes[T.EVAL_ID]} and the "
                         f"walking env {sizes[T.LOCO_EVAL]}")
    return False


def motion_mix(new_files, use_synthetic=True, new_share=0.6, walking=False):
    """Files for the env and kind weights that give each kind the wanted share of episodes.
    The env picks clips in proportion to their length times their kind's weight, so a kind's weight
    is its share divided by its total (usable) length."""
    syn = ([DEFAULT_MOTIONS.resolve()] + ([LOCO_MOTIONS.resolve()] if walking else [])) if use_synthetic else []
    files = list(new_files) + syn
    try:
        lib = MotionLibrary(files)
    except ValueError as e:
        raise SystemExit(f"can't combine the motion files: {e}")
    n_new = MotionLibrary(new_files).n_clips                 # new clips come first in the combined library
    syn_kinds = sorted({str(k) for k in lib.kinds[n_new:]})
    new_kinds = sorted({str(k) for k in lib.kinds[:n_new]})
    shared = [k for k in new_kinds if k in syn_kinds]        # same name as a synthetic kind: can't be weighted apart
    own = [k for k in new_kinds if k not in syn_kinds]
    seconds = Counter()
    for c in range(lib.n_clips):
        if lib.duration(c) >= MIN_CLIP_SECONDS:
            seconds[str(lib.kinds[c])] += lib.duration(c)
    own = [k for k in own if seconds[k] > 0]
    if not own and not shared:
        raise SystemExit(f"none of the new clips is at least {MIN_CLIP_SECONDS} s long")
    share = {}
    if own and syn_kinds:
        rest = [k for k in syn_kinds if seconds[k] > 0]
        total_own = sum(seconds[k] for k in own)
        share.update({k: new_share * seconds[k] / total_own for k in own})
        share.update({k: (1.0 - new_share) / len(rest) for k in rest})
    else:                                                    # nothing to weight apart: every second counts the same
        total = sum(seconds.values())
        share = {k: v / total for k, v in seconds.items()}
    weights = {k: float(share[k] / seconds[k]) for k in share}
    info = {"n_new_clips": n_new, "new_kinds": new_kinds, "shared_kinds": shared, "seconds": dict(seconds),
            "share": share}
    return files, weights, info, lib


def print_mix(new_files, info, lib, use_synthetic):
    print("new motions:")
    for f in new_files:
        m = MotionLibrary(f)
        print(f"   {f}  ({m.n_clips} clips, {sum(m.duration(c) for c in range(m.n_clips)) / 60:.1f} min)")
    if use_synthetic:
        print(f"plus the synthetic library ({lib.n_clips - info['n_new_clips']} clips) so nothing is forgotten")
    if info["shared_kinds"]:
        print(f"note: new clips of kind {info['shared_kinds']} share that kind with the synthetic library, "
              "so they can't get their own share of episodes")
    print("episodes per kind:")
    for k, s in sorted(info["share"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:10s} {100 * s:5.1f} %   ({info['seconds'][k] / 60:.1f} min of clips)")


def dry_run(a, base, files, weights, info, lib, env_id, eval_id=T.EVAL_ID):
    """Check the mix the env really samples, and how the starting policy does on the new clips."""
    env = gym.make(env_id, motion_file=[str(f) for f in files], kind_weights=weights).unwrapped
    rng = np.random.default_rng(0)
    got = Counter(str(lib.kinds[env.lib.sample_clip(rng, weights, MIN_CLIP_SECONDS)]) for _ in range(4000))
    print("sampled by the env (4000 draws): " + ", ".join(f"{k} {100 * n / 4000:.0f} %" for k, n in got.most_common()))
    env.close()

    model = PPO.load(base, device="cpu")
    with open(base + "_vecnormalize.pkl", "rb") as f:
        vecnorm = pickle.load(f)
    ev = gym.make(eval_id, motion_file=[str(f) for f in files]).unwrapped
    new = [c for c in range(info["n_new_clips"]) if lib.duration(c) >= MIN_CLIP_SECONDS][:a.check_clips]
    print(f"starting policy on {len(new)} of your clips (up to 10 s each, from the start of the clip):")
    surv = []
    for c in new:
        obs, _ = ev.reset(seed=0, options={"clip": c, "t0": 0.0})
        errs, info_step = [], {}
        while True:
            action, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
            obs, _, term, trunc, info_step = ev.step(action)
            errs.append(info_step["kp_err"])
            if term or trunc:
                break
        s = float(info_step.get("survival", 1.0))
        surv.append(s)
        end = "fell" if info_step.get("fell") else "lost the pose" if info_step.get("lost") else "ok"
        print(f"   {str(lib.names[c]):30s} {min(lib.duration(c), 10.0):5.1f} s   survival {s:4.2f}   "
              f"pose error {100 * np.mean(errs):4.1f} cm   {end}")
    if surv:
        print(f"mean survival {np.mean(surv):.2f} - fine-tuning should raise this (eval/survival_human in progress.csv)")
    ev.close()


def finetune(a, base, files, weights, info, env_id, steps, eval_id=T.EVAL_ID):
    env_kwargs = {"motion_file": [str(f) for f in files], "kind_weights": weights}
    log_dir = prepare_run_dir(os.path.join(T.LOG_ROOT, a.name))
    venv = T.make_env(env_id, a.n_envs, **env_kwargs)
    venv = VecNormalize.load(base + "_vecnormalize.pkl", venv)
    venv.training, venv.norm_reward = True, True
    model = PPO.load(base, env=venv, device="cpu", learning_rate=T.linear_schedule(*a.lr))
    fmt = ["stdout", "csv"]
    try:
        import tensorboard  # noqa: F401
        fmt.append("tensorboard")
    except ImportError:
        pass
    model.set_logger(configure(log_dir, fmt))
    save_config(log_dir, run=a.name, finetuned_from=os.path.abspath(base), env_id=env_id,
                env_kwargs={**gym.spec(env_id).kwargs, **env_kwargs}, motion_files=[str(f) for f in files],
                episode_share=info["share"], eval_env_id=eval_id, n_envs=a.n_envs, total_timesteps=steps,
                ppo=T.PPO_KW, learning_rate=list(a.lr), curriculum=False,
                video_every_pct=T.VIDEO_EVERY_PCT, checkpoint_every_pct=T.CHECKPOINT_EVERY_PCT,
                eval_episodes_per_kind=a.eval_episodes, mujoco_gl=dict(GL_INFO))

    recorder = PeriodicVideoEvalCallback(eval_id, steps, log_dir, a.name, every_pct=T.VIDEO_EVERY_PCT,
                                         video_seconds_per_kind=T.VIDEO_SECONDS_PER_KIND,
                                         episodes_per_kind=a.eval_episodes,
                                         env_kwargs={"motion_file": [str(f) for f in files]})
    callbacks = [TrackingStatsCallback(),
                 PeriodicCheckpointCallback(steps, log_dir, a.name, every_pct=T.CHECKPOINT_EVERY_PCT),
                 recorder]
    print(f"fine-tuning {base} for {steps:,} steps -> {log_dir}/")
    finished = False
    try:
        model.learn(steps, callback=callbacks, progress_bar=True, reset_num_timesteps=False)
        finished = True
    except KeyboardInterrupt:
        print("Interrupted - saving the current model.")
    model.save(os.path.join(log_dir, a.name))
    venv.save(os.path.join(log_dir, f"{a.name}_vecnormalize.pkl"))
    if not finished:
        try:
            recorder.record_now()
        except Exception as e:                               # noqa: BLE001
            print(f"[eval] final evaluation skipped: {e}")
    recorder.close()
    venv.close()
    print(f"Everything for this run is in {log_dir}/  - try it live: "
          f"python examples/RL/live_pose.py --model {os.path.join(log_dir, a.name)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="model", required=True, help="trained model to start from (with or without .zip)")
    ap.add_argument("--name", required=True, help="name of the new run (logs/<name>/)")
    ap.add_argument("--motions", nargs="+", required=True, help=".npz files, folders or patterns")
    ap.add_argument("--steps", default="10M", help="training steps, e.g. 10M, 500k (default 10M)")
    ap.add_argument("--new-share", type=float, default=0.6, help="share of episodes on the new clips (0-1)")
    ap.add_argument("--no-synthetic", action="store_true", help="train on the new motions only")
    ap.add_argument("--robust", action="store_true", help="add stage 2's pushes, friction changes and sensor noise")
    ap.add_argument("--lr", type=float, nargs=2, default=T.LR_STAGE2, metavar=("START", "END"),
                    help="learning rate, falling linearly from START to END (default 5e-5 1e-5)")
    ap.add_argument("--n-envs", type=int, default=T.N_ENVS, help=f"parallel environments (default {T.N_ENVS})")
    ap.add_argument("--eval-episodes", type=int, default=T.EVAL_EPISODES_PER_KIND,
                    help="evaluation episodes per motion kind")
    ap.add_argument("--dry-run", action="store_true", help="show the mix and test the starting policy, don't train")
    ap.add_argument("--check-clips", type=int, default=20, help="clips tested by --dry-run")
    a = ap.parse_args()
    if not 0.0 < a.new_share <= 1.0:
        ap.error("--new-share must be between 0 and 1")

    base = model_base(a.model)
    new_files = motion_files(a.motions)
    if DEFAULT_MOTIONS.resolve() in new_files:              # e.g. --motions src/G1_tele/assets/motions
        new_files.remove(DEFAULT_MOTIONS.resolve())
        print(f"({DEFAULT_MOTIONS.name} is the synthetic library, not a new motion file: "
              f"{'it is mixed in anyway' if not a.no_synthetic else 'left out (--no-synthetic)'})")
        if not new_files:
            raise SystemExit("no new motion files left")
    walking = is_walking_model(base)
    if walking:
        if not LOCO_MOTIONS.exists():
            raise SystemExit(f"{base} is a walking model but {LOCO_MOTIONS} is missing: "
                             "python examples/RL/make_motions.py loco")
        new_files = [f for f in new_files if f != LOCO_MOTIONS.resolve()]
        print("walking model: fine-tuning in the locomotion env, with the walking clips mixed in")
    files, weights, info, lib = motion_mix(new_files, not a.no_synthetic, a.new_share, walking)
    print_mix(new_files, info, lib, not a.no_synthetic)
    if walking:
        env_id, eval_id = (T.LOCO_ROBUST if a.robust else T.LOCO_TRAIN), T.LOCO_EVAL
    else:
        env_id, eval_id = (T.STAGE2 if a.robust else T.STAGE1), T.EVAL_ID
    if a.dry_run:
        return dry_run(a, base, files, weights, info, lib, env_id, eval_id)
    finetune(a, base, files, weights, info, env_id, parse_steps(a.steps), eval_id)


if __name__ == "__main__":
    main()
