# pose-to-biped

Teaching a simulated Unitree G1 humanoid to copy a person's pose while keeping its balance, with reinforcement learning in MuJoCo. The long-term aim is to drive the robot from a live body tracker. This README covers the simulation and reinforcement learning side; pose detection will be documented separately.

![The simulated G1 following reference poses](assets/following.gif)

One policy drives 23 of the G1's 29 joints (the wrists are held still). It is given target positions for 12 body keypoints (hands, feet, elbows, knees, shoulders and hips) and is rewarded first for staying balanced and then for matching the pose. It is trained with PPO from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) on a library of 120 reference motions.

**Stage 1 results** (60 million steps, log in [`logs/pose_stage1T1/progress.csv`](logs/pose_stage1T1/progress.csv)): test survival rose from 13 % to 89 % of each 10 s test episode, episodes ending in a fall dropped from 34 % to 4.5 %, and the average keypoint error fell from 8.8 cm to 5.6 cm.

## Contents

- [What's in the repository](#whats-in-the-repository)
- [Installation](#installation)
- [Quick start: watch a trained policy](#quick-start-watch-a-trained-policy)
- [Examples](#examples)
  - [Train a policy](#train-a-policy-train_pose_trackingpy)
  - [Make GIFs of checkpoints](#make-gifs-of-checkpoints-gif_makerpy)
  - [Build motion libraries](#build-motion-libraries-make_motionspy)
  - [Drive the G1 with the keyboard](#drive-the-g1-with-the-keyboard-keyboard_controlpy)
  - [Use the environment in your own code](#use-the-environment-in-your-own-code)
- [Troubleshooting](#troubleshooting)
- [Older scripts](#older-scripts)
- [Pose detection](#pose-detection)

## What's in the repository

Only the parts used for reinforcement learning are listed here.

```
src/G1_tele/                    the Python package (installed by `pip install -e .`)
  envs/
    pose_tracking.py            G1-pose-tracking: the whole-body pose-tracking environment
    whole_body.py               29-joint G1 base: PD control, joint armature, torque limits
    walker.py                   G1-walker: the 12 leg motors only, for Unitree's walking policy
  motion/
    library.py                  reference-motion files (.npz): load, save, merge
    synthetic.py                generates balanced whole-body motions with inverse kinematics
    retarget.py                 turns tracker keypoints into a full robot pose
    kinematics.py               forward kinematics and the keypoint definitions
  policies/
    pose_tracking.py            the actor-critic network used for training
    motion.pt, g1.yaml          Unitree's pretrained walking policy and its settings
  assets/
    g1_description/             Unitree G1 models (MuJoCo .xml, .urdf) and meshes
    motions/synthetic.npz       the default motion library: 120 clips, 42.8 min
examples/RL/
  train_pose_tracking.py        train, test and watch policies
  run_logging.py                logs, evaluation videos, checkpoints and the curriculum
  gif_maker.py                  turns checkpoints into GIFs (no window needed)
  make_motions.py               builds or extends the motion library
  show_policy.py                minimal example of calling visualise() from your own script
examples/keyboard_control.py    drive the G1 with the arrow keys using the pretrained walking policy
logs/
  pose_stage1/checkpoints/      a stage 1 checkpoint from 30 % of training (18 M steps)
  pose_stage1T1/progress.csv    the full stage 1 training log
```

## Installation

You need Python 3.10 to 3.12. Training has been run on Linux. macOS and Windows should work too; see [Troubleshooting](#troubleshooting). Training runs on the CPU, so you don't need a GPU. A GPU is only used to render evaluation videos on machines without a screen.

```bash
git clone https://github.com/shepai/pose-to-biped.git
cd pose-to-biped
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

# Optional: CPU-only PyTorch, a much smaller download if you have no NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# The G1_tele package, plus numpy, torch, mujoco, gymnasium, opencv, tqdm and rich
pip install -e .

# Training, videos and GIFs
pip install stable-baselines3 imageio imageio-ffmpeg pillow

# Optional: live training curves
pip install tensorboard
```

**Install with `-e` (editable mode).** The robot model, meshes, motion library and walking policy sit inside `src/G1_tele/`, but they aren't declared as package data. A plain `pip install .` copies the code without them, and the environments then can't find their files.

**Linux without a screen** (a server, or over SSH): evaluation videos are rendered with EGL, which needs the system OpenGL libraries.

```bash
sudo apt install libegl1 libgl1
# fallback if EGL doesn't work: sudo apt install libosmesa6 && pip install PyOpenGL
```

**Check the installation.** Run these commands from the repository root:

```bash
python examples/RL/train_pose_tracking.py check
```

This builds the environment, runs 10 episodes with no corrections from the policy, and prints the sizes and simulation speed. It should start with:

```
obs (240,) (actor sees the first 221), action (23,), 120 clips, 42.8 min of motion
```

The untrained robot falls within a couple of seconds; that's expected. Then check that videos can be recorded:

```bash
python examples/RL/train_pose_tracking.py video
```

This tries each OpenGL backend (`glfw`, `egl`, `osmesa`) and the video writers, then records a short clip to `logs/video_test/`. If videos don't work, it tells you which `MUJOCO_GL` setting to use. Training works without videos.

## Quick start: watch a trained policy

The repository includes a checkpoint from 30 % of stage 1 (18 million steps):

```bash
# Opens a MuJoCo window. Red spheres are the left-side targets, blue the right.
python examples/RL/train_pose_tracking.py play logs/pose_stage1/checkpoints/pose_stage1_030pct_18.0M

# No screen? Record a GIF instead.
python examples/RL/gif_maker.py logs/pose_stage1/checkpoints/pose_stage1_030pct_18.0M --out stage1_30pct.gif
```

Run all commands from the repository root: runs are written to `logs/` in the current folder.

## Examples

### Train a policy (`train_pose_tracking.py`)

```bash
python examples/RL/train_pose_tracking.py <mode>
```

| Mode | What it does |
|---|---|
| `check` | Builds the environment and runs 10 zero-action episodes (about 20 s, no training). |
| `video` | Tests which OpenGL backend can render, tests the video writers and records a test clip. |
| `train` | **Stage 1**: 60 M steps on the motion library, written to `logs/pose_stage1/`. |
| `train2` | **Stage 2**: 40 M more steps starting from the end of stage 1, with pushes on the pelvis (up to 60 N), floor friction varied from 0.6 to 1.3 times normal, and sensor noise. Written to `logs/pose_stage2/`. |
| `play [model]` | Watches stage 1's final model, or any model or checkpoint you give it. |
| `play2 [model]` | The same for stage 2. |

- **Run time:** stage 1 took about 10.4 hours on our machine with 16 environments in parallel (about 1,600 steps per second). Set `N_ENVS` to about the number of CPU cores you have.
- **Stopping early:** Ctrl-C stops training safely. The current model is saved and one last evaluation is recorded.
- **Stage 2's starting point:** stage 2 starts from `logs/pose_stage1/pose_stage1.zip`, the final stage 1 model, which only exists once stage 1 has finished or been stopped. The repository includes only the 30 % checkpoint.
- **Old runs are never overwritten:** if `logs/<run>/` already has files, it is renamed to `logs/<run>_<date-time>/` before the new run starts. This includes the `logs/pose_stage1/` folder that ships with the repository.

#### How training works

- **The curriculum:** training starts with motions where both feet stay planted (`upper`: arms and waist; `whole`: squats, leans and twists). One-leg balancing (`balance`) is added once every motion kind already in training survives 80 % of its episodes, and stepping (`step`) after that. In our stage 1 run, balancing was added at 1.9 M steps and stepping at 46.0 M.
- **The reward** is 20 % for staying up, 40 % for balance and 40 % for pose accuracy. Balance means:
  - keeping the capture point inside the feet
  - pelvis tilt and height
  - little wobble
  - correct foot contacts

  Pose accuracy covers keypoint positions, joint angles and joint speeds. The weights are `REWARD_WEIGHTS` in `src/G1_tele/envs/pose_tracking.py`.
- **The critic sees more than the actor:** it also gets 19 values only the simulator knows, such as true velocity, centre of mass, foot contacts, friction and pushes. The actor (the part you deploy) only sees what the real robot can measure.

#### Settings

These are at the top of `train_pose_tracking.py`:

| Setting | Default | Meaning |
|---|---|---|
| `COMMON` | see file | Environment options shared by every stage: motion file, robot xml, keypoint set, reward weights |
| `N_ENVS` | 16 | Environments simulated in parallel |
| `PPO_KW` | `n_steps=512, batch_size=2048, n_epochs=5, gamma=0.995, target_kl=0.03` | PPO settings |
| `LR_STAGE1`, `LR_STAGE2` | `(2e-4, 3e-5)`, `(5e-5, 1e-5)` | Learning rate, falling linearly from the first value to the second |
| `CURRICULUM`, `CURRICULUM_START`, `CURRICULUM_THRESHOLD` | `True`, 2, 0.8 | Curriculum on/off, motion kinds unlocked at the start, survival needed to unlock the next kind |
| `ASSIST` | 0.0 | Optional stabilising push on the pelvis that fades out early in training (off by default) |
| `VIDEO_EVERY_PCT` | 5 | Evaluate and record a video every 5 % of the run |
| `EVAL_EPISODES_PER_KIND` | 2 | Fixed test episodes per motion kind at each evaluation |
| `CHECKPOINT_EVERY_PCT` | 10 | Save a model backup every 10 % of the run |
| `LOG_ROOT` | `"logs"` | Where runs are written, relative to the current folder |

#### What a run writes

```
logs/<run>/
  config.json                        every setting used, and the start time
  progress.csv                       the training log (one row per PPO update)
  eval.csv                           one row per evaluation: survival and errors, overall and per motion kind
  curriculum.json                    which motion kinds are unlocked, and their sampling weights
  videos/<run>_005pct_3.0M.mp4       one clip per evaluation, a few seconds of each motion kind
  checkpoints/<run>_010pct_6.0M      model backup (see the note below)
  checkpoints/<run>_010pct_6.0M_vecnormalize.pkl
  <run>.zip, <run>_vecnormalize.pkl  the final model
  events.out.tfevents.*              only if tensorboard is installed
```

- **Checkpoint file names:** checkpoints are saved without a `.zip` extension. Stable-Baselines3 only adds `.zip` when the name has no other extension, and the `.0M` at the end counts as one. Pass checkpoint paths exactly as they appear on disk; `play` and `gif_maker.py` accept them with or without `.zip`.
- **The `_vecnormalize.pkl` file:** a model always needs the `_vecnormalize.pkl` saved next to it, which holds the running statistics used to normalise observations. `play`, `gif_maker.py` and `load_policy()` load it for you.

#### Reading `progress.csv`

| Column | Meaning |
|---|---|
| `eval/survival` | Fraction of the fixed test episodes the robot survives (1.0 = never falls). The most comparable number across a run. |
| `eval/survival_<kind>` | The same for each motion kind: `upper`, `whole`, `balance`, `step`. |
| `eval/kp_err_cm` | Average keypoint error in the test episodes. |
| `curriculum/level` | Motion kinds unlocked: 2 = `upper` + `whole`, 3 adds `balance`, 4 adds `step`. |
| `curriculum/survival_<kind>` | Survival per kind in training; the next kind unlocks when all of these are at least 0.8. |
| `track/end_fell` | Fraction of training episodes that ended with a fall. |
| `track/end_lost` | Fraction that ended because the pose was more than 30 cm off on average. |
| `track/cp_margin_cm` | How far the capture point stays inside the feet; above 0 means balanced. |
| `track/kp_err_cm`, `track/hand_err_cm`, `track/foot_err_cm` | Pose error per step in training. |
| `track/reward_balance`, `track/reward_tracking` | The two halves of the reward. |
| `rollout/ep_rew_mean` | Standard SB3 episode reward. It dips every time the curriculum adds harder motions, so the columns above are more useful. |

The evaluation columns are only filled in on rows where an evaluation ran (every 5 %). With `tensorboard` installed you can watch the same values live: `tensorboard --logdir logs`.

#### Watching a policy from your own script

`visualise()` in `train_pose_tracking.py` opens the viewer for any model. `examples/RL/show_policy.py` shows how to call it:

```python
from train_pose_tracking import visualise, STAGE1
visualise(STAGE1, "logs/pose_stage1/checkpoints/pose_stage1_030pct_18.0M")
```

### Make GIFs of checkpoints (`gif_maker.py`)

`gif_maker.py` renders off-screen, so it also works over SSH and on servers.

```bash
# one checkpoint, written next to it as <checkpoint>.gif
python examples/RL/gif_maker.py logs/pose_stage1/checkpoints/pose_stage1_030pct_18.0M

# every checkpoint of a run (quote the pattern; checkpoint names end in "M")
python examples/RL/gif_maker.py "logs/pose_stage1/checkpoints/*M"

# only one-leg balance motions, 15 s, larger
python examples/RL/gif_maker.py <checkpoint> --kind balance --seconds 15 --width 640 --height 480

# choose the output file
python examples/RL/gif_maker.py <checkpoint> --out progress/30pct.gif
```

| Option | Default | Meaning |
|---|---|---|
| `--seconds` | 10 | Length of each GIF |
| `--kind` | all | Only play one motion kind: `upper`, `whole`, `balance` or `step` |
| `--fps` | 25 | GIF frame rate (the simulation runs at 50 Hz) |
| `--width`, `--height` | 480, 360 | Image size |
| `--seed` | 0 | Which episodes are played; the same seed gives every checkpoint the same episodes |
| `--out` | next to the checkpoint | Output file (only with a single checkpoint) |
| `--no-label` | off | Don't print the checkpoint name in the corner |
| `--env` | `G1-pose-eval` | Environment to play in |

Each GIF shows the target spheres, and the script prints how each episode ended (fell, lost track or clip ended) with its keypoint error. Without `--out`, an existing GIF next to the checkpoint is overwritten.

### Build motion libraries (`make_motions.py`)

The policy learns from reference motions: sequences of full robot poses that are physically possible. The default library, `src/G1_tele/assets/motions/synthetic.npz`, has 30 clips each of four kinds:

| Kind | Motions | Feet |
|---|---|---|
| `upper` | arm and waist movements | both planted |
| `whole` | squats, leans, twists and weight shifts, with arm movements | both planted |
| `balance` | knee raises and foot reaches on one leg | one lifted, then put back |
| `step` | steps to new spots, and turns | one lifted, then placed somewhere new |

```bash
# generate synthetic motions
# (without --out this overwrites the default library, so pass --out to keep it)
python examples/RL/make_motions.py synth --clips-per-kind 30 --seconds 20 --out my_synthetic.npz

# keypoints from a tracker -> full-body reference clip
python examples/RL/make_motions.py keypoints path/to/clip.npz --fps 30 --out my_clip.npz

# old hand-tracking data (T, 2, 3 hand positions relative to the shoulders) -> clips
python examples/RL/make_motions.py hands path/to/hands.npy --fps 30 --out my_hands.npz

# combine several libraries into one
python examples/RL/make_motions.py merge combined.npz my_synthetic.npz my_clip.npz
```

- **Keypoint input** for `keypoints` is an `.npz` with one `(T, 3)` array per keypoint name. Use the names `left_hand`, `right_hand`, `left_foot` and `right_foot`, plus optionally the elbows, knees, hips and shoulders. Positions must already be in the robot's heading frame and at robot scale.
- **Retargeting** solves inverse kinematics for every frame (about 2 ms per frame) and prints the mean and maximum keypoint error of each clip.
- **Training on your library:** set `motion_file` in `COMMON` in `train_pose_tracking.py` to its path, or to a list of paths.
- **Curriculum and your clips:** clips made with `keypoints` get the kind `keypoints`, which the curriculum never locks, so they are trained on from the start. Clips made with `hands` get the kind `upper`.

### Drive the G1 with the keyboard (`keyboard_control.py`)

Unitree's pretrained walking policy controls the 12 leg joints of the G1, and you steer it with the keyboard. It needs a screen.

```bash
python examples/keyboard_control.py                    # uses src/G1_tele/policies/motion.pt
python examples/keyboard_control.py --policy path/to/other_policy.pt
```

Click the viewer window first, then:

| Key | Action |
|---|---|
| ↑ / ↓ | faster forward / backward (±0.2 m/s per press) |
| ← / → | turn left / right (±0.3 rad/s per press) |
| `,` / `.` | step sideways left / right |
| Space | stop |
| Backspace | reset |

### Use the environment in your own code

Importing `G1_tele` registers the environments with Gymnasium:

| Id | What it is |
|---|---|
| `G1-pose-tracking` | Whole-body pose tracking (the environment trained above) |
| `G1-walker` | The 12 leg motors only, with no reward; used by `keyboard_control.py` |

```python
import gymnasium as gym
import numpy as np
import G1_tele                                   # registers the G1-* environments

env = gym.make("G1-pose-tracking")
obs, info = env.reset(seed=0)
print(env.observation_space.shape, env.action_space.shape)   # (240,) (23,)

for _ in range(500):
    action = np.zeros(env.action_space.shape)    # zero = follow the reference pose, no correction
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(info["clip_kind"], "survived", info["survival"], "fell" if info["fell"] else "")
        obs, info = env.reset()
```

#### Options

Pass these to `gym.make("G1-pose-tracking", ...)`:

| Option | Default | Meaning |
|---|---|---|
| `keypoints` | `"full"` | Keypoints in the reward: `"minimal"` (hands and feet), `"limbs"` (adds elbows and knees), `"full"` (adds shoulders and hips), or a list of names |
| `obs_keypoints` | same as `keypoints` | Keypoints the policy sees, e.g. `"minimal"` for a tracker that only gives hands and feet |
| `motion_file` | the synthetic library | A `.npz` motion library, or a list of them |
| `xml_path` | `g1_29dof.xml` | Robot model |
| `residual` | `True` | Actions are corrections added to the reference joint angles (otherwise, offsets from a default pose) |
| `reward_weights` | `None` | Change reward weights, e.g. `dict(kp=0.3, cp=0.1)`; see `REWARD_WEIGHTS` |
| `episode_seconds` | 10.0 | Maximum episode length |
| `max_kp_err` | 0.30 | End the episode if the average keypoint error exceeds this (m) |
| `push_force`, `friction_range`, `obs_noise`, `init_noise` | 0, `None`, 0, 0.03 | Pushes (N), floor friction range, sensor noise and start-pose noise |
| `kind_weights` | `None` | How often each motion kind is sampled, e.g. `{"upper": 1, "whole": 1}` |
| `gravity_ff` | `True` | Add the torque needed to hold the reference pose against gravity |
| `privileged_obs` | `True` | Append the simulator-only values the critic uses |

#### Observations, actions and info

- **Actions** are 23 values between −1 and 1, one per driven joint. They are added to the reference joint angles, then a PD controller at 500 Hz turns the result into motor torque. The policy acts at 50 Hz.
- **Observations** start with the part the actor uses, the first `env.unwrapped.n_actor_obs` values (221 with the defaults). That part holds:
  - the IMU readings
  - joint angles and speeds
  - the last action
  - the target keypoints and how far the robot's own keypoints are from them
  - the reference joint angles

  The rest is simulator-only data for the critic. Positions are measured relative to the pelvis and the direction the robot faces.
- **Step info** includes `kp_err`, `hand_err`, `foot_err` (m), `cp_margin`, `fell` and `lost`. At the end of an episode it also includes `clip_kind`, `survival` (the fraction of the possible episode survived) and `episode_kp_err`.
- **Drawing the targets:** `env.unwrapped.draw_targets(scene)` adds the target spheres to a MuJoCo viewer or renderer scene.

#### Loading a trained policy

This is what `load_policy()` in `train_pose_tracking.py` does:

```python
import gymnasium as gym
import G1_tele
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ckpt = "logs/pose_stage1/checkpoints/pose_stage1_030pct_18.0M"
venv = VecNormalize.load(ckpt + "_vecnormalize.pkl", DummyVecEnv([lambda: gym.make("G1-pose-tracking")]))
venv.training, venv.norm_reward = False, False   # use the saved statistics, don't update them
model = PPO.load(ckpt, device="cpu")

obs = venv.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)
```

The environment options must match the ones the model was trained with. `train_pose_tracking.py` uses the defaults.

## Troubleshooting

**No evaluation videos, or "VIDEOS ARE OFF" in the training log.** Run `python examples/RL/train_pose_tracking.py video` and use the backend it recommends, e.g. `MUJOCO_GL=egl python examples/RL/train_pose_tracking.py train`. The scripts switch to EGL automatically when there is no display at all.

**Training stalls while recording videos over SSH.** With X forwarding (`DISPLAY=localhost:10.0`), MuJoCo's default `glfw` backend renders, but every frame crosses the network, and one evaluation video can take many minutes. Start training with `MUJOCO_GL=egl` to render on the remote machine instead.

**Videos are saved as GIFs or `_frames.npz` instead of `.mp4`.** Install `imageio` and `imageio-ffmpeg`. The scripts fall back to OpenCV, then GIF, then raw frames.

**`ImportError: cannot import name ... from 'run_logging'`.** Python picked up an older `run_logging.py`. Make sure `train_pose_tracking.py`, `run_logging.py` and `gif_maker.py` are the versions in `examples/RL/` and sit in the same folder.

**macOS: the viewer doesn't open.** MuJoCo's viewer needs `mjpython` on macOS. Run `play` and `keyboard_control.py` with `mjpython` instead of `python`. EGL isn't available on macOS, so videos use `glfw`.

**`FileNotFoundError` for robot files or motions.** The package was installed without `-e`. Run `pip install -e .` from the repository root.

## Older scripts

These scripts are from earlier experiments. They are not part of the G1 pipeline above and may not run as they are:

| Script | What it was |
|---|---|
| `examples/hand_tracking.py`, `examples/recorder.py` | First G1 attempt: Unitree's walking policy on the legs and a second policy for the arms. It needs `G1_tele/envs/hand_tracking.py`, which isn't in the repository, so it doesn't run at the moment. |
| `examples/RL_pose_train.py`, `examples/play_RL.py`, `examples/record_RL.py` | Earlier Unitree H1 experiments using `sim/`. They have hard-coded paths to a `Robots/` folder that is no longer in the repository. |
| `models/ppo/` | Checkpoints from those H1 runs. |

## Pose detection

The `pose/` folder turns video into body keypoints with MediaPipe. It will be documented here later, along with connecting a live tracker to the trained policy.

## Built on

- [MuJoCo](https://github.com/google-deepmind/mujoco) for simulation and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for the environment interface.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO.
- G1 models and the pretrained walking policy from [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym).
- Per-motor armature values from [BeyondMimic's G1 configuration](https://github.com/HybridRobotics/whole_body_tracking).
- The training approach follows [DeepMimic](https://arxiv.org/abs/1804.02717) and [H2O](https://arxiv.org/abs/2403.04436), with the [capture point](https://portfolio.erau.edu/en/publications/capture-point-a-step-toward-humanoid-push-recovery/) as the balance measure.