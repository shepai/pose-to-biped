"""
examples/train_pose_tracking.py  -  whole-body pose tracking for the G1 with Stable-Baselines3 PPO

    python train_pose_tracking.py check      # 20 s sanity check of the env (no training)
    python train_pose_tracking.py train      # stage 1: track the motion library, no disturbances
    python train_pose_tracking.py train2     # stage 2: resume stage 1 + pushes, friction, sensor noise
    python train_pose_tracking.py play       # watch stage 1 (play2 for stage 2) - spheres are the targets
    python train_pose_tracking.py play logs/pose_stage1/checkpoints/pose_stage1_050pct_30.0M.zip

Everything a run produces goes to logs/<run>/ (see run_logging.py):
    progress.csv, eval.csv, config.json, videos/ (every VIDEO_EVERY_PCT %), checkpoints/ (every
    CHECKPOINT_EVERY_PCT %), and the final <run>.zip + <run>_vecnormalize.pkl. Starting a run whose
    folder already exists moves the old folder to logs/<run>_<date-time> instead of overwriting it.

What to watch in logs/<run>/progress.csv (or TensorBoard if installed):
    track/ep_len_s       seconds survived per episode   (starts ~0.9 s, should climb towards 10 s)
    track/kp_err_cm      mean keypoint error per step   (starts ~10 cm, good policies reach 3-5 cm)
    track/fall_rate      fraction of episodes ending in a fall / lost tracking
    track/assist         balance-assist strength if you enable it (ep_len dips while it fades; judge
                         the policy on the numbers after it reaches 0)
    rollout/ep_rew_mean  is still logged, but it mixes episode length and quality, so it is noisy;
                         track/reward_per_step is the cleaner number.
    eval/*               the same 5 fixed episodes, deterministic policy, unassisted - the most
                         comparable number across the run (also in eval.csv next to each video)

The env ids are registered at the TOP LEVEL of this file because SubprocVecEnv workers need them in
every process.
"""
import os
import sys
import time

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import G1_tele                                                   # noqa: F401  (registers G1-* ids)
from G1_tele.policies.hand_tracking import HandTrackingPolicy    # plain actor-critic MLP, see that file
from run_logging import (PeriodicCheckpointCallback, PeriodicVideoEvalCallback, TrackingStatsCallback,
                         prepare_run_dir, save_config)

# ------------------------------------------------------------------ config (top level on purpose)
ENTRY = "G1_tele.envs.pose_tracking:PoseTrackingEnv"
COMMON = dict(
    motion_file=None,          # None -> G1_tele/assets/motions/synthetic.npz; or a list of .npz files
    xml_path=None,             # None -> G1_tele/assets/g1_description/g1_29dof.xml
    keypoints="full",          # reward keypoints: "minimal" (hands+feet), "limbs", "full", or a list
    obs_keypoints=None,        # keypoints the policy sees (None = same); e.g. "minimal" for a 4-point tracker
    residual=True,             # PD target = reference joints + action (needs IK at deploy time)
)
N_ENVS = 16                    # ~ number of auto cores
# Optional balance assist: a stabilising push on the pelvis (towards the reference height and tilt)
# that fades linearly to zero over the first ASSIST_STEPS of stage 1. Off by default (in short tests it
# did not beat plain training); try ASSIST = 1.0 if track/ep_len_s is still flat after ~10M steps.
ASSIST, ASSIST_STEPS = 0.0, 12_000_000
STAGE1, STAGE2 = "G1-pose-stage1", "G1-pose-stage2"
register(id=STAGE1, entry_point=ENTRY, kwargs={**COMMON, "assist": ASSIST,
                                                "assist_anneal_steps": ASSIST_STEPS // N_ENVS})
register(id="G1-pose-eval", entry_point=ENTRY, kwargs={**COMMON})   # no assist / noise: videos, play
register(id=STAGE2, entry_point=ENTRY, kwargs={**COMMON, "push_force": 60.0, "friction_range": (0.6, 1.3),
                                                "obs_noise": 0.01, "init_noise": 0.05})

EVAL_ID = "G1-pose-eval"

POLICY_KW = dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]), log_std_init=-1.0)
PPO_KW = dict(n_steps=256, batch_size=1024, n_epochs=5, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
              ent_coef=0.0, vf_coef=0.5, max_grad_norm=1.0)

LOG_ROOT = "logs"
VIDEO_EVERY_PCT = 5            # evaluate + film every 5 % of the run (and at 0 %)
VIDEO_SECONDS = 8.0            # length of each video
EVAL_EPISODES = 5              # fixed episodes per evaluation (only the first VIDEO_SECONDS are filmed)
CHECKPOINT_EVERY_PCT = 10      # back up model + normalisation stats every 10 % of the run


def linear_schedule(start, end):
    return lambda progress_remaining: end + (start - end) * progress_remaining


def make_env(env_name, n_envs, **env_kwargs):
    return make_vec_env(env_name, n_envs=n_envs, env_kwargs=env_kwargs or None,
                        vec_env_cls=SubprocVecEnv if n_envs > 1 else None)


def run(env_name, run_name, total_timesteps, n_envs=N_ENVS, init_run=None):
    log_dir = prepare_run_dir(os.path.join(LOG_ROOT, run_name))

    venv = make_env(env_name, n_envs)
    if init_run:                                             # stage 2: reuse weights + normalisation stats
        src = os.path.join(LOG_ROOT, init_run, init_run)
        venv = VecNormalize.load(f"{src}_vecnormalize.pkl", venv)
        venv.training, venv.norm_reward = True, True
        lr = (1e-4, 3e-5)
        model = PPO.load(src, env=venv, device="auto", learning_rate=linear_schedule(*lr))
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=PPO_KW["gamma"])
        lr = (3e-4, 5e-5)
        model = PPO(HandTrackingPolicy, venv, policy_kwargs=POLICY_KW, learning_rate=linear_schedule(*lr),
                    device="auto", verbose=1, **PPO_KW)
    fmt = ["stdout", "csv"]
    try:
        import tensorboard  # noqa: F401
        fmt.append("tensorboard")
    except ImportError:
        pass
    model.set_logger(configure(log_dir, fmt))
    save_config(log_dir, run=run_name, env_id=env_name, env_kwargs=gym.spec(env_name).kwargs,
                eval_env_id=EVAL_ID, n_envs=n_envs, total_timesteps=total_timesteps, init_run=init_run,
                ppo=PPO_KW, policy=POLICY_KW, learning_rate=lr, video_every_pct=VIDEO_EVERY_PCT,
                checkpoint_every_pct=CHECKPOINT_EVERY_PCT, eval_episodes=EVAL_EPISODES)

    recorder = PeriodicVideoEvalCallback(EVAL_ID, total_timesteps, log_dir, run_name, every_pct=VIDEO_EVERY_PCT,
                                         video_seconds=VIDEO_SECONDS, eval_episodes=EVAL_EPISODES)
    callbacks = [TrackingStatsCallback(),
                 PeriodicCheckpointCallback(total_timesteps, log_dir, run_name, every_pct=CHECKPOINT_EVERY_PCT),
                 recorder]
    finished = False
    try:
        model.learn(total_timesteps, callback=callbacks, progress_bar=True,
                    reset_num_timesteps=init_run is None)
        finished = True
    except KeyboardInterrupt:
        print("Interrupted - saving the current model.")
    model.save(os.path.join(log_dir, run_name))
    venv.save(os.path.join(log_dir, f"{run_name}_vecnormalize.pkl"))
    if not finished:                                         # final video/eval of what was saved
        try:
            recorder.record_now()
        except Exception as e:                               # noqa: BLE001
            print(f"[eval] final evaluation skipped: {e}")
    recorder.close()
    venv.close()
    print(f"Everything for this run is in {log_dir}/")


def load_policy(model_path, env_name):
    """Model + a VecNormalize wrapper holding the training statistics (needed to normalise obs).
    model_path: logs/<run>/<run>(.zip) or any checkpoint zip; its _vecnormalize.pkl sits next to it."""
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    venv = make_env(env_name, 1, assist=0.0)                 # never assist when evaluating
    venv = VecNormalize.load(f"{base}_vecnormalize.pkl", venv)
    venv.training, venv.norm_reward = False, False
    return PPO.load(base, device="auto"), venv


def visualise(env_name, model_path, seconds=None):
    import mujoco
    import mujoco.viewer
    model, venv = load_policy(model_path, env_name)
    raw = venv.venv.envs[0].unwrapped
    obs = venv.reset()
    with mujoco.viewer.launch_passive(raw.model, raw.data) as viewer:
        t0 = time.time()
        while viewer.is_running() and (seconds is None or time.time() - t0 < seconds):
            tic = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(action)
            if done[0]:
                print(f"episode end - mean keypoint error {100 * info[0].get('episode_kp_err', np.nan):.1f} cm")
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                raw.draw_targets(viewer.user_scn)            # red = left, blue = right
            viewer.sync()
            time.sleep(max(0.0, raw.dt - (time.time() - tic)))
    venv.close()


def check():
    env = gym.make(STAGE1, assist=0.0).unwrapped
    print(f"obs {env.observation_space.shape}, action {env.action_space.shape}, "
          f"{env.lib.n_clips} clips, {sum(env.lib.duration(c) for c in range(env.lib.n_clips)) / 60:.1f} min of motion")
    lens, t0, n = [], time.time(), 0
    for ep in range(10):
        env.reset(seed=ep)
        for k in range(500):
            _, r, term, trunc, info = env.step(np.zeros(env.action_space.shape))
            n += 1
            if term or trunc:
                break
        lens.append(k + 1)
    print(f"untrained (zero action) episode length: {np.mean(lens) * env.dt:.2f} s (it has to learn balance), "
          f"{n / (time.time() - t0):.0f} env steps/s in one process")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "check":
        check()
    elif mode == "train":
        run(STAGE1, "pose_stage1", 60_000_000)
    elif mode == "train2":
        run(STAGE2, "pose_stage2", 40_000_000, init_run="pose_stage1")
    elif mode in ("play", "play2"):
        default = os.path.join(LOG_ROOT, "pose_stage1", "pose_stage1") if mode == "play" else \
            os.path.join(LOG_ROOT, "pose_stage2", "pose_stage2")
        visualise(EVAL_ID, sys.argv[2] if len(sys.argv) > 2 else default)
    else:
        raise SystemExit(__doc__)
