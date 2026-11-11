"""
Unitree H1, 19 joints (MuJoCo Menagerie's unitree_h1/scene.xml; fetch it with examples/get_robots.py).

Differences from the G1 that matter here:
  * one ankle joint per leg (pitch only): the sole cannot be held level sideways, so the motion
    tools leave the foot's roll (mostly) free, and the foot is narrow - standing on one leg is much
    harder. With flat feet the pelvis can't move sideways over a foot at all (every leg joint that
    could do it would also tilt the foot), so the references roll the stance foot onto its edge to
    shift the weight, as the real robot has to; on one leg a rolled sole edge can sit up to ~2 cm
    below the floor in the reference,
  * the waist only turns (no roll / pitch), and there are no wrists or hands: the "hand" keypoint
    is the end of the forearm.

PD gains and the leg default pose: unitree_rl_gym (legged_gym/envs/h1/h1_config.py). Torque limits
are read from the model's actuators. The arm default pose (slightly out, elbows a little below
level) is this package's choice.
"""
from .spec import ASSETS, SIDES, Joint, RobotSpec

#        xml name suffix   slot part      q0    kp   kd
_LEG = [("hip_yaw",        "hip_yaw",     0.0, 150, 2),
        ("hip_roll",       "hip_roll",    0.0, 150, 2),
        ("hip_pitch",      "hip_pitch",  -0.1, 150, 2),
        ("knee",           "knee",        0.3, 200, 4),
        ("ankle",          "ankle_pitch", -0.2, 40, 2)]
_ARM = [("shoulder_pitch", "shoulder_pitch", 0.0, 150, 2),
        ("shoulder_roll",  "shoulder_roll",  0.1, 150, 2),      # right: -0.1
        ("shoulder_yaw",   "shoulder_yaw",   0.0, 150, 2),
        ("elbow",          "elbow",          0.3, 100, 2)]

_joints = [Joint("torso", "waist_yaw", 0.0, 300, 6)]
for side in SIDES:
    for name, part, q0, kp, kd in _LEG:
        _joints.append(Joint(f"{side}_{name}", f"{side}_{part}", q0, kp, kd))
    for name, part, q0, kp, kd in _ARM:
        if part == "shoulder_roll" and side == "right":
            q0 = -q0
        _joints.append(Joint(f"{side}_{name}", f"{side}_{part}", q0, kp, kd))

_kp = {}
for side in SIDES:
    _kp.update({
        f"{side}_hand": (f"{side}_elbow_link", (0.28, 0.0, -0.015)),          # end of the forearm
        f"{side}_elbow": (f"{side}_elbow_link", (0.0, 0.0, 0.0)),
        f"{side}_shoulder": (f"{side}_shoulder_roll_link", (0.0, 0.0, 0.0)),
        f"{side}_hip": (f"{side}_hip_pitch_link", (0.0, 0.0, 0.0)),
        f"{side}_knee": (f"{side}_knee_link", (0.0, 0.0, 0.0)),
        f"{side}_foot": (f"{side}_ankle_link", (0.0, 0.0, 0.0)),              # ankle
        f"{side}_toe": (f"{side}_ankle_link", (0.14, 0.0, -0.056)),
    })
_kp["head"] = ("torso_link", (0.05, 0.0, 0.68))

H1 = RobotSpec(
    name="h1",
    xml_path=ASSETS / "robots" / "unitree_h1" / "scene.xml",
    joints=tuple(_joints),
    keypoints=_kp,
    feet=("left_ankle_link", "right_ankle_link"),
    root_body="pelvis",
    explore={"waist_yaw": (-0.6, 0.6),
             "shoulder_pitch": (-2.6, 0.7),       # - = raise forward
             "shoulder_roll": (-0.15, 1.7),       # + = outward (left arm)
             "shoulder_yaw": (-0.9, 0.9),
             "elbow": (-0.7, 1.55)},              # 0 = forearm level, 1.57 = straight
    notes="No ankle roll: one-leg balance and stepping are much harder than on the G1.",
)
