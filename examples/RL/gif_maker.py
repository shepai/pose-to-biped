"""
examples/RL/gif_maker.py  -  record policy checkpoints to GIFs (off-screen, no window; works over SSH)

    python gif_maker.py logs/pose_stage1/checkpoints/pose_stage1_030pct_18.0M
    python gif_maker.py logs/pose_stage1/checkpoints/*.zip                     # one GIF per checkpoint
    python gif_maker.py <checkpoint> --kind balance --seconds 15 --width 640 --height 480
    python gif_maker.py <checkpoint> --out progress/30pct.gif

Each GIF is written next to its checkpoint (same name, .gif) unless --out is given. Every checkpoint
plays the same episodes (same --seed), so GIFs from different points in training are comparable.
Target spheres are drawn (red = left, blue = right) and the checkpoint name is printed in the corner.
"""
import argparse
import glob
import os

import numpy as np

from train_pose_tracking import EVAL_ID, load_policy     # also picks the OpenGL backend (EGL over SSH)
import mujoco                                              # noqa: E402  (after the backend is chosen)


def _label(path):
    name = os.path.basename(path[:-4] if path.endswith(".zip") else path)
    parts = name.split("_")                                # pose_stage1_030pct_18.0M -> "stage1  30%  18.0M steps"
    try:
        pct = next(p for p in parts if p.endswith("pct"))
        steps = next(p for p in parts if p.endswith("M"))
        stage = next((p for p in parts if p.startswith("stage")), "")
        return f"{stage}  {int(pct[:-3])}%  {steps} steps"
    except StopIteration:
        return name


def record_frames(model, venv, seconds=10.0, fps=25, width=480, height=360, kind=None, seed=0):
    """Run the deterministic policy and return (frames, episode summaries)."""
    raw = venv.venv.envs[0].unwrapped
    if kind is not None:
        if kind not in raw.clip_kinds:
            raise SystemExit(f"--kind must be one of {raw.clip_kinds}")
        raw.set_kind_weights({kind: 1.0})
    m = raw.model
    w = min(width, int(m.vis.global_.offwidth))
    h = min(height, int(m.vis.global_.offheight))
    renderer = mujoco.Renderer(m, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = raw.pelvis_id
    cam.distance, cam.azimuth, cam.elevation = 2.6 * raw.kin.scale, 135.0, -15.0     # further back for bigger robots

    venv.seed(seed)
    obs = venv.reset()
    every = max(1, round(1.0 / (raw.dt * fps)))            # 50 Hz control -> every 2nd step at 25 fps
    frames, episodes, t_ep = [], [], 0.0
    try:
        for step in range(int(seconds / raw.dt)):
            if step % every == 0:
                renderer.update_scene(raw.data, camera=cam)
                raw.draw_targets(renderer.scene)
                frames.append(renderer.render().copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(action)
            t_ep += raw.dt
            if done[0]:
                end = "fell" if info[0].get("fell") else "lost track" if info[0].get("lost") else "clip ended"
                episodes.append(f"{info[0].get('clip_kind', '?')}: {t_ep:.1f} s, {end}, "
                                f"kp err {100 * info[0].get('episode_kp_err', np.nan):.1f} cm")
                t_ep = 0.0
    finally:
        renderer.close()
    return frames, episodes


def save_gif(frames, path, fps, label=None):
    """Adaptive-palette GIF via Pillow, optionally with a text label in the top-left corner."""
    from PIL import Image, ImageDraw
    adaptive = getattr(getattr(Image, "Palette", Image), "ADAPTIVE")
    imgs = []
    for f in frames:
        im = Image.fromarray(np.asarray(f, np.uint8))
        if label:
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 8 + 7 * len(label), 20], fill=(0, 0, 0))
            d.text((5, 4), label, fill=(255, 255, 255))
        imgs.append(im.convert("P", palette=adaptive, colors=256))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(round(1000 / fps)), loop=0,
                 optimize=True)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="model path(s), with or without .zip (globs are fine)")
    p.add_argument("--env", default=EVAL_ID, help="env id (default: the unassisted eval env)")
    p.add_argument("--robot", default=None, help="robot to film (default: the model's first robot)")
    p.add_argument("--seconds", type=float, default=10.0, help="length of each GIF")
    p.add_argument("--kind", default=None, help="only play one motion kind: upper, whole, balance, step")
    p.add_argument("--fps", type=int, default=25, help="GIF frame rate (the sim runs at 50 Hz)")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--seed", type=int, default=0, help="same seed = same episodes for every checkpoint")
    p.add_argument("--out", default=None, help="output .gif (only with a single checkpoint)")
    p.add_argument("--no-label", action="store_true", help="don't print the checkpoint name on the GIF")
    a = p.parse_args()

    paths = []
    for c in a.checkpoints:
        hits = sorted(glob.glob(c)) or [c]
        paths += [h for h in hits if not h.endswith("_vecnormalize.pkl")]
    if a.out and len(paths) > 1:
        raise SystemExit("--out only works with a single checkpoint")

    for path in paths:
        base = path[:-4] if path.endswith(".zip") else path
        out = a.out or f"{base}{'_' + a.robot if a.robot else ''}{'_' + a.kind if a.kind else ''}.gif"
        print(f"recording {os.path.basename(base)} ({a.seconds:.0f} s, MUJOCO_GL="
              f"{os.environ.get('MUJOCO_GL', 'glfw')}) ...", flush=True)
        model, venv = load_policy(path, a.env, a.robot)
        try:
            frames, episodes = record_frames(model, venv, a.seconds, a.fps, a.width, a.height, a.kind, a.seed)
        finally:
            venv.close()
        save_gif(frames, out, a.fps, None if a.no_label else _label(path))
        for e in episodes:
            print(f"   episode {e}")
        print(f"   -> {out}  ({len(frames)} frames, {os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main() #python gif_maker.py /home/dexter/Documents/GitHub/pose-to-biped/logs/pose_stage1/checkpoints/pose_stage1_030pct_18.0M
