"""
Download the MuJoCo models of the robots that don't ship with the package (everything but the G1)
from MuJoCo Menagerie, into G1_tele/assets/robots/<folder>/:

    python examples/get_robots.py            # every robot
    python examples/get_robots.py h1 t1      # some of them

Needs git. The models are fetched at a fixed Menagerie commit, so the joint names, keypoint bodies and
measurements in G1_tele/robots/*.py stay valid. Each folder keeps the model's own LICENSE.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from G1_tele.robots import ROBOTS

MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"


def menagerie_folder(spec):
    """The Menagerie folder a robot's model comes from (None for models shipped with the package)."""
    p = Path(spec.xml_path)
    return p.parent.name if p.parent.parent.name == "robots" else None


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{r.stderr.strip()}")


def main(names):
    specs = [ROBOTS[n] for n in (names or sorted(ROBOTS))]
    todo = {menagerie_folder(s): s for s in specs if menagerie_folder(s)}
    if not todo:
        print("nothing to download (the G1 ships with the package)")
        return
    if shutil.which("git") is None:
        raise SystemExit("git not found. Download https://github.com/google-deepmind/mujoco_menagerie/archive/"
                         f"{COMMIT}.zip and copy the folders {sorted(todo)} into "
                         f"{Path(next(iter(todo.values())).xml_path).parent.parent}")
    with tempfile.TemporaryDirectory() as tmp:
        print(f"fetching {', '.join(sorted(todo))} from MuJoCo Menagerie ({COMMIT[:8]}) ...", flush=True)
        run(["git", "init", "-q"], tmp)
        run(["git", "remote", "add", "origin", MENAGERIE], tmp)
        run(["git", "sparse-checkout", "set", "--no-cone", *[f"/{f}/" for f in todo]], tmp)
        run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT], tmp)
        run(["git", "checkout", "-q", "FETCH_HEAD"], tmp)
        for folder, spec in sorted(todo.items()):
            dst = Path(spec.xml_path).parent
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(Path(tmp) / folder, dst, ignore=shutil.ignore_patterns("*.png", "CHANGELOG.md"))
            ok = Path(spec.xml_path).exists()
            print(f"  {spec.name}: {dst}" + ("" if ok else f"  (but {Path(spec.xml_path).name} is missing!)"))
    print("next:  python examples/RL/check_robot.py <robot>   then   python examples/RL/make_motions.py synth --robot <robot>")


if __name__ == "__main__":
    unknown = [n for n in sys.argv[1:] if n not in ROBOTS]
    if unknown:
        raise SystemExit(f"unknown robot(s) {unknown}; known: {sorted(ROBOTS)}")
    main(sys.argv[1:])
