"""
Download the models of the robots that don't ship with the package (everything but the G1) into
G1_tele/assets/robots/<folder>/:

    python examples/get_robots.py            # every robot
    python examples/get_robots.py h1 icub    # some of them

Needs git. The H1, T1 and N1 come from MuJoCo Menagerie; the iCub from the iCub team's URDF
(robotology/icub-models), converted to MuJoCo here (G1_tele.robots.urdf_to_mjcf). Both are fetched at
fixed commits, so the joint names, keypoint bodies and measurements in G1_tele/robots/*.py stay valid.
Each folder keeps the model's own LICENSE.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from G1_tele.robots import ROBOTS

MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"
ICUB_MODELS = "https://github.com/robotology/icub-models.git"
ICUB_COMMIT = "dca8c618c264be70ce285bad38040411725e3551"
ICUB_URDF = "iCub/robots/iCubGazeboV2_7/model.urdf"
CONVERTED = {"icub"}                                   # robots built from a URDF rather than Menagerie


def menagerie_folder(spec):
    """The Menagerie folder a robot's model comes from (None for models shipped with the package)."""
    if spec.name in CONVERTED:
        return None
    p = Path(spec.xml_path)
    return p.parent.name if p.parent.parent.name == "robots" else None


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{r.stderr.strip()}")


def sparse_fetch(repo, commit, paths, tmp):
    """Check out only `paths` of repo at commit into tmp."""
    run(["git", "init", "-q"], tmp)
    run(["git", "remote", "add", "origin", repo], tmp)
    run(["git", "sparse-checkout", "set", "--no-cone", *paths], tmp)
    run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], tmp)
    run(["git", "checkout", "-q", "FETCH_HEAD"], tmp)


def get_menagerie(todo):
    with tempfile.TemporaryDirectory() as tmp:
        print(f"fetching {', '.join(sorted(todo))} from MuJoCo Menagerie ({COMMIT[:8]}) ...", flush=True)
        sparse_fetch(MENAGERIE, COMMIT, [f"/{f}/" for f in todo], tmp)
        for folder, spec in sorted(todo.items()):
            dst = Path(spec.xml_path).parent
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(Path(tmp) / folder, dst, ignore=shutil.ignore_patterns("*.png", "CHANGELOG.md"))
            ok = Path(spec.xml_path).exists()
            print(f"  {spec.name}: {dst}" + ("" if ok else f"  (but {Path(spec.xml_path).name} is missing!)"))


def get_icub(spec):
    from G1_tele.robots.icub import armature_for
    from G1_tele.robots.urdf_to_mjcf import add_rest_excludes, write_mjcf
    dst = Path(spec.xml_path).parent
    with tempfile.TemporaryDirectory() as tmp:
        print(f"fetching the iCub 2.7 URDF from robotology/icub-models ({ICUB_COMMIT[:8]}) ...", flush=True)
        sparse_fetch(ICUB_MODELS, ICUB_COMMIT, ["/LICENSE", f"/{ICUB_URDF}"], tmp)
        urdf = Path(tmp) / ICUB_URDF
        meshes = sorted(set(re.findall(r'filename="package://(iCub/meshes/[^"]+)"', urdf.read_text())))
        run(["git", "sparse-checkout", "set", "--no-cone", "/LICENSE", f"/{ICUB_URDF}", *[f"/{m}" for m in meshes]], tmp)
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "meshes").mkdir(parents=True)
        for m in meshes:
            shutil.copy(Path(tmp) / m, dst / "meshes" / Path(m).name)
        shutil.copy(Path(tmp) / "LICENSE", dst / "LICENSE")
        shutil.copy(urdf, dst / "model.urdf")
        joints = re.findall(r'<joint name="([^"]+)" type="revolute"', urdf.read_text())
        arm = {j: armature_for(j) for j in joints if armature_for(j) is not None}
        write_mjcf(urdf, dst, "icub", base_rotation=np.diag([-1.0, -1.0, 1.0]), joint_armature=arm)
    (dst / "README.md").write_text(
        "# iCub 2.7 for MuJoCo\n\n"
        f"Converted by pose-to-biped's examples/get_robots.py from `{ICUB_URDF}` of\n"
        f"https://github.com/robotology/icub-models (commit {ICUB_COMMIT}), licensed CC BY-SA 4.0\n"
        "(see LICENSE). Changes: fixed links merged into the moving links they hang from, the robot\n"
        "turned to face +x, rotor inertia added, torque motors on every joint, a floor (scene.xml).\n")
    print(f"  icub: {dst} ({len(meshes)} meshes)")
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("  (MuJoCo isn't installed, so contacts between links that touch in the standing pose\n"
              "   weren't excluded; run this again where MuJoCo is installed)")
        return
    from G1_tele.motion.kinematics import Kinematics
    k = Kinematics("icub")
    pairs = add_rest_excludes(dst / "icub.xml", dst / "scene.xml", k.standing_qpos())
    print(f"  excluded {len(pairs)} body pair(s) whose collision shapes overlap in the standing pose")


def main(names):
    specs = [ROBOTS[n] for n in (names or sorted(ROBOTS))]
    todo = {menagerie_folder(s): s for s in specs if menagerie_folder(s)}
    converted = [s for s in specs if s.name in CONVERTED]
    if not todo and not converted:
        print("nothing to download (the G1 ships with the package)")
        return
    if shutil.which("git") is None:
        raise SystemExit("git not found. Download https://github.com/google-deepmind/mujoco_menagerie/archive/"
                         f"{COMMIT}.zip and copy the folders {sorted(todo)} into "
                         f"{Path(specs[0].xml_path).parent.parent} (the iCub needs git)")
    if todo:
        get_menagerie(todo)
    for spec in converted:
        get_icub(spec)
    print("next:  python examples/RL/check_robot.py <robot>   then   python examples/RL/make_motions.py synth --robot <robot>")


if __name__ == "__main__":
    unknown = [n for n in sys.argv[1:] if n not in ROBOTS]
    if unknown:
        raise SystemExit(f"unknown robot(s) {unknown}; known: {sorted(ROBOTS)}")
    main(sys.argv[1:])
