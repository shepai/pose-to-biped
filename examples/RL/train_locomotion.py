"""
examples/RL/train_locomotion.py  -  one policy that walks, turns and copies a pose while keeping its balance

    # once: the walking and turning clips (-> src/G1_tele/assets/motions/locomotion.npz)
    python examples/RL/make_motions.py loco

    # continue from a policy that already balances and copies poses (recommended)
    python examples/RL/train_locomotion.py --from logs/pose_stage1T1/pose_stage1T1 --name loco1
    # ... with your own recordings mixed in
    python examples/RL/train_locomotion.py --from logs/pose_human1/pose_human1 --name loco1 \\
        --motions src/G1_tele/assets/motions/human
    # or the whole ladder from scratch
    python examples/RL/train_locomotion.py --scratch --name loco1
    # then harden it: pushes, slippery floors, sensor noise
    python examples/RL/train_locomotion.py --from logs/loco1/loco1 --name loco1_robust --robust

    python examples/RL/train_locomotion.py --from ... --name ... --dry-run    # check everything, don't train
    python examples/RL/train_locomotion.py play logs/loco1/loco1 --kind walk  # watch it (any kind, or all)

The curriculum: balance, then pose, then motion
-----------------------------------------------
Clip kinds unlock one at a time, each once the robot survives at least 80 % of its episodes on every
kind unlocked so far (the same rule as stage 1, extended):

    upper, whole     feet planted: balance while the arms, waist and pelvis move
    balance, step    standing on one leg, single steps
    turn             turning on the spot
    walk             walking forwards, backwards and sideways, starting and stopping
    loco             walking and turning while the arms and waist move
    human_walk       your own walking recordings (record_motion.py --walk), if you add them
    lafan            Unitree's retargeted LAFAN1 walking (import_lafan.py), if you add it

From --from, everything up to `turn` is unlocked at the start, because the policy can already do
the first four. Survival now includes staying on course: ending up more than 0.5 m or about 45 degrees
away from the reference root ends the episode, so standing still on a walking clip doesn't pass.
Your own pose recordings (kind "human") are never locked: they are mixed in from the start.

Starting from a trained pose policy
-----------------------------------
The locomotion env observes 6 more numbers (the reference's root velocity and turning rate, and how
far the robot is from the reference root). The trained network gets zero weights for them, so it
starts out doing exactly what it did before; its observation statistics are kept, and the new
entries' statistics are measured from a short run. Action noise below --min-std is raised to it, so
there is some exploration for the new skill. The learning rate (1e-4 falling to 2e-5) sits between
stage 1's and stage 2's.

What to watch in logs/<name>/progress.csv
-----------------------------------------
    curriculum/level           7 = everything unlocked
    curriculum/survival_<kind> training survival per kind
    eval/survival_<kind>       the fixed evaluation episodes, per kind
    eval/vel_err_m_s           how far the robot's speed is from the reference's
    track/end_off_course       episodes ended by drifting off course (vs end_fell)
    track/root_err_cm, track/yaw_err_deg   distance and heading error to the reference root
Videos (logs/<name>/videos/) show a green dot where the reference root is.

Then try it live: python examples/RL/live_pose.py --model logs/<name>/<name>  (it detects that the
model walks and follows your position and heading too).
"""
import argparse
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_tracking as T          # noqa: E402  (OpenGL set-up, env registration, PPO settings)
from finetune import model_base, motion_files, parse_steps  # noqa: E402
from run_logging import (GL_INFO, CurriculumCallback, PeriodicCheckpointCallback, PeriodicVideoEvalCallback,  # noqa: E402
                         TrackingStatsCallback, prepare_run_dir, save_config)

import gymnasium as gym                  # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402
from stable_baselines3 import PPO        # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from G1_tele.motion.library import DEFAULT_MOTIONS, LOCO_MOTIONS, MotionLibrary, default_motions  # noqa: E402,F401
from G1_tele.policies.pose_tracking import PoseTrackingPolicy  # noqa: E402
from G1_tele.policies.widen import floor_log_std, widen_state_dict, widen_stats  # noqa: E402

ORDER = ("upper", "whole", "balance", "step", "turn", "walk", "loco", "human_walk", "lafan")
START_FROM_TRAINED, START_SCRATCH = 5, 2            # kinds of ORDER unlocked at the start
THRESHOLD = 0.8
LR_FROM, LR_SCRATCH, LR_CONTINUE = (1e-4, 2e-5), T.LR_STAGE1, T.LR_STAGE2
STEPS_FROM, STEPS_SCRATCH = "40M", "150M"
MIN_STD = 0.15
STATS_STEPS = 3000                                  # env steps used to measure the new inputs' statistics


# ------------------------------------------------------------------ motions
def motion_list(extra, robot="g1"):
    base = [default_motions(robot).resolve(), default_motions(robot, loco=True).resolve()]
    for f, what in zip(base, ("synth", "loco")):
        if not f.exists():
            raise SystemExit(f"{f} is missing - make it first:  python examples/RL/make_motions.py {what} --robot {robot}")
    more = [f for f in (motion_files(extra) if extra else []) if f not in base]
    try:
        lib = MotionLibrary(base + more, robot=robot)
    except ValueError as e:
        raise SystemExit(f"can't combine the motion files: {e}")
    return base + more, lib


def print_motions(files, lib, start):
    sec = Counter()
    for c in range(lib.n_clips):
        if lib.duration(c) >= 1.5:
            sec[str(lib.kinds[c])] += lib.duration(c)
    print("motions: " + ", ".join(str(f) for f in files))
    for k in ORDER:
        if k in sec:
            when = "from the start" if ORDER.index(k) < start else "unlocked later"
            print(f"   {k:10s} {sec[k] / 60:5.1f} min   {when}")
    for k in sorted(set(sec) - set(ORDER)):
        print(f"   {k:10s} {sec[k] / 60:5.1f} min   always on (yours)")


# ------------------------------------------------------------------ model set-up
def _sizes(model):
    n_obs = int(model.observation_space.shape[0])
    n_actor = model.policy_kwargs.get("actor_obs_dim") or n_obs
    return n_obs, int(n_actor)


def _env_sizes(venv):
    get = lambda name: venv.get_attr(name, indices=[0])[0]      # noqa: E731
    return int(venv.observation_space.shape[0]), int(get("n_actor_obs")), int(get("n_pose_obs"))


def new_input_stats(env_id, env_kwargs, n_pose, n_actor, steps=STATS_STEPS):
    """Mean/var of the locomotion inputs, from a short run with zero actions (the reference's own motion)."""
    env = gym.make(env_id, **env_kwargs).unwrapped
    xs, (obs, _) = [], env.reset(seed=0)
    for i in range(steps):
        obs, _, term, trunc, _ = env.step(np.zeros(env.action_space.shape, np.float32))
        xs.append(obs[n_pose:n_actor])
        if term or trunc:
            obs, _ = env.reset(seed=i + 1)
    env.close()
    xs = np.array(xs, float)
    return xs.mean(0), xs.var(0) + 1e-4


def load_vecnormalize(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_from(base, venv, env_id, env_kwargs, lr, min_std):
    """PPO model + VecNormalize for the locomotion env, starting from a trained model.
    A pose-only model is widened; a locomotion model is simply continued. Returns (model, venv, how)."""
    old = PPO.load(base, device="cpu")
    old_obs, old_actor = _sizes(old)
    new_obs, new_actor, n_pose = _env_sizes(venv)
    if (old_obs, old_actor) == (new_obs, new_actor):
        venv = VecNormalize.load(base + "_vecnormalize.pkl", venv)
        venv.training, venv.norm_reward = True, True
        return PPO.load(base, env=venv, device="cpu", learning_rate=T.linear_schedule(*lr)), venv, "continued"
    if old_actor != n_pose or old_obs - old_actor != new_obs - new_actor:
        raise SystemExit(f"{base} expects {old_obs} observations (actor {old_actor}); the locomotion env has "
                         f"{new_obs} (actor {new_actor}, pose part {n_pose}). Was it trained with other env settings?")
    old_vn = load_vecnormalize(base + "_vecnormalize.pkl")
    mu_new, var_new = new_input_stats(env_id, env_kwargs, n_pose, new_actor)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=old_vn.clip_obs,
                        clip_reward=old_vn.clip_reward, gamma=old_vn.gamma, epsilon=old_vn.epsilon)
    mu, var = widen_stats(old_vn.obs_rms.mean, old_vn.obs_rms.var, old_actor, old_obs, new_actor, new_obs,
                          mu_new, var_new)
    venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = mu, var, old_vn.obs_rms.count
    venv.ret_rms = old_vn.ret_rms
    kw = {**old.policy_kwargs, "actor_obs_dim": new_actor}
    model = PPO(PoseTrackingPolicy, venv, policy_kwargs=kw, learning_rate=T.linear_schedule(*lr), device="cpu",
                verbose=1, **T.PPO_KW)
    sd = widen_state_dict({k: v.detach().cpu().numpy() for k, v in old.policy.state_dict().items()},
                          old_actor, old_obs, new_actor, new_obs)
    std0 = np.exp(sd["log_std"])
    sd["log_std"] = floor_log_std(sd["log_std"], min_std)
    cur = model.policy.state_dict()
    model.policy.load_state_dict({k: torch.as_tensor(v, dtype=cur[k].dtype) for k, v in sd.items()})
    print(f"widened {base}: actor inputs {old_actor} -> {new_actor} (new ones start with zero weight); "
          f"action noise {std0.mean():.3f} -> {np.exp(sd['log_std']).mean():.3f} (mean std, floor {min_std})")
    return model, venv, "widened"


def check_same_actions(base, model, venv, env_id, env_kwargs, steps=60):
    """The widened policy must act exactly like the original on the same states."""
    old = PPO.load(base, device="cpu")
    old_vn = load_vecnormalize(base + "_vecnormalize.pkl")
    old_obs, old_actor = _sizes(old)
    new_obs, new_actor, n_pose = _env_sizes(venv)
    env = gym.make(env_id, **env_kwargs).unwrapped
    obs, _ = env.reset(seed=3)
    worst = 0.0
    for i in range(steps):
        o_old = np.concatenate([obs[:n_pose], obs[new_actor:]])
        a_old, _ = old.predict(old_vn.normalize_obs(o_old), deterministic=True)
        a_new, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
        worst = max(worst, float(np.max(np.abs(np.asarray(a_old) - np.asarray(a_new)))))
        obs, _, term, trunc, _ = env.step(a_new)
        if term or trunc:
            obs, _ = env.reset(seed=100 + i)
    env.close()
    return worst


def evaluate(model, venv, env_kwargs, episodes_per_kind=4, seconds=10.0):
    """Survival per kind of the deterministic policy, on the locomotion eval env."""
    env = gym.make(T.LOCO_EVAL, **env_kwargs).unwrapped
    lib, rng = env.lib, np.random.default_rng(0)
    out = {}
    for kind in [k for k in ORDER if k in env.clip_kinds] + [k for k in env.clip_kinds if k not in ORDER]:
        clips = [c for c in range(lib.n_clips) if str(lib.kinds[c]) == kind and lib.duration(c) >= 2.0]
        if not clips:
            continue
        surv, ends, verr = [], Counter(), []
        for i in range(episodes_per_kind):
            c = clips[int(rng.integers(len(clips)))]
            obs, _ = env.reset(seed=i, options={"clip": c, "t0": float(rng.uniform(0, max(0.0, lib.duration(c) - seconds)))})
            for _ in range(int(seconds / env.dt)):
                a, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
                obs, _, term, trunc, info = env.step(a)
                verr.append(info["vel_err"])
                if term or trunc:
                    break
            surv.append(info.get("survival", 1.0))
            ends["fell" if info["fell"] else "off course" if info.get("off_course") else "lost the pose"
                 if info["lost"] else "ok"] += 1
        out[kind] = (float(np.mean(surv)), dict(ends), float(np.mean(verr)))
    env.close()
    return out


# ------------------------------------------------------------------ train / dry run / play
def robots_and_layout(a, base):
    """Robots to train / play on and the layout: --robot / --layout, else what the starting model was
    trained with (its run's config.json), else the G1."""
    cfg = T.run_config(base) if base else {}
    robots = T.parse_robots(a.robot) if getattr(a, "robot", None) else (cfg.get("robots") or ["g1"])
    layout = getattr(a, "layout", None) or cfg.get("layout") or ("universal" if len(robots) > 1 else "native")
    if len(robots) > 1 and layout != "universal":
        raise SystemExit("several robots need --layout universal")
    if len(robots) > 1 and getattr(a, "motions", None):
        raise SystemExit("--motions works with one robot at a time (motion files belong to one robot)")
    return robots, layout


def train(a):
    base = None if a.scratch else model_base(a.model)
    robots, layout = robots_and_layout(a, base)
    files, lib = motion_list(a.motions, robots[0])
    for r in robots[1:]:
        motion_list(None, r)                                 # every robot needs its motions
    # single-env helpers (dry run, widening, evaluation) use the first robot; the training envs all of them
    env_kwargs = {"motion_file": [str(f) for f in files], "robot": robots[0], "layout": layout}
    vec_kwargs = {"motion_file": env_kwargs["motion_file"]} if len(robots) == 1 else {}
    env_id = T.LOCO_ROBUST if a.robust else T.LOCO_TRAIN
    start = a.start_level or (START_SCRATCH if a.scratch else START_FROM_TRAINED)
    print_motions(files, lib, start)
    steps = parse_steps(a.steps or (STEPS_SCRATCH if a.scratch else STEPS_FROM))
    print(f"robots: {', '.join(robots)} ({layout} layout)")

    if a.dry_run:
        venv = T.make_env(env_id, 1, robots=robots[:1], layout=layout, **vec_kwargs)
    else:
        log_dir = prepare_run_dir(os.path.join(T.LOG_ROOT, a.name))
        venv = T.make_env(env_id, a.n_envs, robots=robots, layout=layout, **vec_kwargs)
    prev = None
    if a.scratch:
        lr = a.lr or LR_SCRATCH
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=T.PPO_KW["gamma"])
        n_actor = venv.get_attr("n_actor_obs", indices=[0])[0]
        model = PPO(PoseTrackingPolicy, venv, policy_kwargs={**T.POLICY_KW, "actor_obs_dim": n_actor},
                    learning_rate=T.linear_schedule(*lr), device="cpu", verbose=1, **T.PPO_KW)
        how = "from scratch"
    else:
        old_obs, _ = _sizes(PPO.load(base, device="cpu"))
        continuing = old_obs == venv.observation_space.shape[0]
        lr = a.lr or (LR_CONTINUE if continuing else LR_FROM)
        model, venv, how = build_from(base, venv, env_id, env_kwargs, lr, a.min_std)
        cj = os.path.join(os.path.dirname(base), "curriculum.json")
        prev = cj if os.path.exists(cj) else None

    if a.dry_run:
        print(f"\nenv {env_id}: {venv.observation_space.shape[0]} observations, actor sees "
              f"{venv.get_attr('n_actor_obs', indices=[0])[0]}; model {how}")
        if how == "widened":
            worst = check_same_actions(base, model, venv, env_id, env_kwargs)
            print(f"widened vs original policy on the same states: largest action difference {worst:.2e} "
                  f"({'identical' if worst < 1e-4 else 'NOT identical - please report this'})")
        if not a.scratch:
            print(f"\nstarting policy, {a.eval_episodes} episodes of up to 10 s per kind:")
            for kind, (s, ends, ve) in evaluate(model, venv, env_kwargs, a.eval_episodes).items():
                print(f"   {kind:10s} survival {s:4.2f}   speed error {ve:4.2f} m/s   {ends}")
        venv.close()
        return

    fmt = ["stdout", "csv"]
    try:
        import tensorboard  # noqa: F401
        fmt.append("tensorboard")
    except ImportError:
        pass
    model.set_logger(configure(log_dir, fmt))
    save_config(log_dir, run=a.name, start=how, started_from=os.path.abspath(base) if base else None,
                env_id=env_id, env_kwargs={**gym.spec(env_id).kwargs, **env_kwargs}, eval_env_id=T.LOCO_EVAL,
                robots=robots, layout=layout,
                n_envs=a.n_envs, total_timesteps=steps, ppo=T.PPO_KW, learning_rate=list(lr),
                min_std=None if a.scratch else a.min_std, curriculum=dict(order=ORDER, start=start, threshold=THRESHOLD,
                                                                          state_from=prev),
                video_every_pct=T.VIDEO_EVERY_PCT, checkpoint_every_pct=T.CHECKPOINT_EVERY_PCT,
                eval_episodes_per_kind=a.eval_episodes, mujoco_gl=dict(GL_INFO))
    multi = len(robots) > 1
    recorders = [PeriodicVideoEvalCallback(T.LOCO_EVAL, steps, log_dir, a.name, every_pct=T.VIDEO_EVERY_PCT,
                                           video_seconds_per_kind=T.VIDEO_SECONDS_PER_KIND,
                                           episodes_per_kind=a.eval_episodes, tag=r if multi else None,
                                           env_kwargs=env_kwargs if not multi else T.robot_kwargs(T.LOCO_EVAL, r, layout))
                 for r in robots]
    callbacks = [TrackingStatsCallback(),
                 CurriculumCallback(log_dir, order=ORDER, start=start, threshold=THRESHOLD, state_from=prev,
                                    per_second=True),
                 PeriodicCheckpointCallback(steps, log_dir, a.name, every_pct=T.CHECKPOINT_EVERY_PCT),
                 *recorders]
    print(f"training {a.name} ({how}) for {steps:,} steps -> {log_dir}/")
    finished = False
    try:
        model.learn(steps, callback=callbacks, progress_bar=True, reset_num_timesteps=True)
        finished = True
    except KeyboardInterrupt:
        print("Interrupted - saving the current model.")
    model.save(os.path.join(log_dir, a.name))
    venv.save(os.path.join(log_dir, f"{a.name}_vecnormalize.pkl"))
    if not finished:
        for recorder in recorders:
            try:
                recorder.record_now()
            except Exception as e:                           # noqa: BLE001
                print(f"[eval] final evaluation skipped: {e}")
    for recorder in recorders:
        recorder.close()
    venv.close()
    print(f"Everything for this run is in {log_dir}/  - try it live: "
          f"python examples/RL/live_pose.py --model {os.path.join(log_dir, a.name)}")


def play(a):
    import mujoco
    import mujoco.viewer
    base = model_base(a.model)
    robots, layout = robots_and_layout(a, base)
    files, _ = motion_list(a.motions, robots[0])
    kw = {"motion_file": [str(f) for f in files]}
    if a.kind:
        kw["kind_weights"] = {a.kind: 1.0}
    venv = T.make_env(T.LOCO_EVAL, 1, robots=robots[:1], layout=layout, **kw)
    venv = VecNormalize.load(base + "_vecnormalize.pkl", venv)
    venv.training, venv.norm_reward = False, False
    model = PPO.load(base, device="cpu")
    raw = venv.venv.envs[0].unwrapped
    obs = venv.reset()
    with mujoco.viewer.launch_passive(raw.model, raw.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING     # follow the robot as it walks
        viewer.cam.trackbodyid = raw.pelvis_id
        viewer.cam.distance, viewer.cam.elevation = 3.0, -15.0
        t0 = time.time()
        while viewer.is_running() and (a.seconds is None or time.time() - t0 < a.seconds):
            tic = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(action)
            if done[0]:
                i = info[0]
                end = "fell" if i.get("fell") else "drifted off course" if i.get("off_course") else \
                    "lost the pose" if i.get("lost") else "end of clip"
                print(f"episode end ({i.get('clip_kind', '?')}): {end}, mean pose error "
                      f"{100 * i.get('episode_kp_err', np.nan):.1f} cm")
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                raw.draw_targets(viewer.user_scn)            # red/blue = targets, green = reference root
            viewer.sync()
            time.sleep(max(0.0, raw.dt - (time.time() - tic)))
    venv.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        ap = argparse.ArgumentParser(prog="train_locomotion.py play")
        ap.add_argument("model")
        ap.add_argument("--kind", default=None, help="only this clip kind (turn, walk, loco, step, ...)")
        ap.add_argument("--motions", nargs="*", default=None, help="extra motion files the model was trained with")
        ap.add_argument("--seconds", type=float, default=None)
        ap.add_argument("--robot", default=None, help="robot to play on (default: the model's first robot)")
        ap.add_argument("--layout", choices=["native", "universal"], default=None)
        return play(ap.parse_args(sys.argv[2:]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="model", help="trained model to start from (with or without .zip)")
    src.add_argument("--scratch", action="store_true", help="train from scratch through the whole curriculum")
    ap.add_argument("--name", required=True, help="name of the new run (logs/<name>/)")
    ap.add_argument("--robot", default=None,
                    help="robot, or a comma list for one shared policy (default: what --from was trained on, else g1)")
    ap.add_argument("--layout", choices=["native", "universal"], default=None,
                    help="default: the --from model's, else native for one robot and universal for several")
    ap.add_argument("--motions", nargs="*", default=None, help="extra .npz files or folders (e.g. your recordings)")
    ap.add_argument("--steps", default=None, help=f"training steps (default {STEPS_FROM} with --from, {STEPS_SCRATCH} from scratch)")
    ap.add_argument("--robust", action="store_true", help="pushes, friction changes and sensor noise (stage 2's)")
    ap.add_argument("--lr", type=float, nargs=2, default=None, metavar=("START", "END"),
                    help=f"learning rate (default {LR_FROM} from a pose model, {LR_CONTINUE} continuing a "
                         f"walking model, {LR_SCRATCH} from scratch)")
    ap.add_argument("--min-std", type=float, default=MIN_STD, help=f"floor for the action noise (default {MIN_STD})")
    ap.add_argument("--start-level", type=int, default=None,
                    help=f"kinds of {', '.join(ORDER)} unlocked at the start (default {START_FROM_TRAINED} with --from, "
                         f"{START_SCRATCH} from scratch)")
    ap.add_argument("--n-envs", type=int, default=T.N_ENVS, help=f"parallel environments (default {T.N_ENVS})")
    ap.add_argument("--eval-episodes", type=int, default=T.EVAL_EPISODES_PER_KIND,
                    help="evaluation episodes per kind (--dry-run: per kind for the starting-policy test)")
    ap.add_argument("--dry-run", action="store_true", help="set everything up and test the starting policy, don't train")
    a = ap.parse_args()
    if a.dry_run and a.eval_episodes == T.EVAL_EPISODES_PER_KIND:
        a.eval_episodes = 4
    train(a)


if __name__ == "__main__":
    main()
