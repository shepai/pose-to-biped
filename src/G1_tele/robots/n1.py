"""
Fourier N1, 23 joints (MuJoCo Menagerie's fourier_n1/scene.xml; fetch it with examples/get_robots.py).

Close to the G1 in layout: the base is the pelvis (base_link), the waist only turns, and each arm has
one wrist joint (it turns the forearm, the G1's wrist roll), which is held still. The ankle rolls
before it pitches, and the foot's collision shape is a mesh, so the sole is measured from that mesh.

PD gains and the leg default pose: Fourier's fourier_aurora_sdk (python/example/fouriern1/demo_walk.py).
Torque limits are read from the model's actuators. The arm default pose is this package's choice.
"""
from .spec import ASSETS, SIDES, Joint, RobotSpec

#        xml name               slot part       q0       kp   kd
_LEG = [("hip_pitch_joint",    "hip_pitch",   -0.2468, 180, 10.0),
        ("hip_roll_joint",     "hip_roll",     0.0,    120, 10.0),
        ("hip_yaw_joint",      "hip_yaw",      0.0,     90,  8.0),
        ("knee_pitch_joint",   "knee",         0.5181, 120,  8.0),
        ("ankle_roll_joint",   "ankle_roll",   0.0,     45,  2.5),
        ("ankle_pitch_joint",  "ankle_pitch", -0.2408,  45,  2.5)]
_ARM = [("shoulder_pitch_joint", "shoulder_pitch", 0.0,  90, 8.0, False),
        ("shoulder_roll_joint",  "shoulder_roll",  0.15, 45, 2.5, False),     # right: -0.15
        ("shoulder_yaw_joint",   "shoulder_yaw",   0.0,  45, 2.5, False),
        ("elbow_pitch_joint",    "elbow",          0.3,  45, 2.5, False),
        ("wrist_yaw_joint",      "wrist_roll",     0.0,  45, 2.5, True)]      # turns the forearm; held still

_joints = [Joint("waist_yaw_joint", "waist_yaw", 0.0, 90, 8.0)]
for side in SIDES:
    for name, part, q0, kp, kd in _LEG:
        _joints.append(Joint(f"{side}_{name}", f"{side}_{part}", q0, kp, kd))
    for name, part, q0, kp, kd, locked in _ARM:
        if part == "shoulder_roll" and side == "right":
            q0 = -q0
        _joints.append(Joint(f"{side}_{name}", f"{side}_{part}", q0, kp, kd, locked=locked))

_kp = {}
for side in SIDES:
    _kp.update({
        f"{side}_hand": (f"{side}_end_effector_link", (0.0, 0.0, 0.0)),
        f"{side}_elbow": (f"{side}_lower_arm_pitch_link", (0.0, 0.0, 0.0)),
        f"{side}_shoulder": (f"{side}_upper_arm_roll_link", (0.0, 0.0, 0.0)),
        f"{side}_hip": (f"{side}_thigh_roll_link", (0.0, 0.0, 0.0)),
        f"{side}_knee": (f"{side}_shank_pitch_link", (0.0, 0.0, 0.0)),
        f"{side}_foot": (f"{side}_foot_pitch_link", (0.0, 0.0, 0.0)),         # ankle
    })
_kp["head"] = ("torso_link", (0.0, 0.0, 0.19))

N1 = RobotSpec(
    name="n1",
    xml_path=ASSETS / "robots" / "fourier_n1" / "scene.xml",
    joints=tuple(_joints),
    keypoints=_kp,
    feet=("left_foot_pitch_link", "right_foot_pitch_link"),
    root_body="base_link",
    explore={"waist_yaw": (-0.6, 0.6),
             "shoulder_pitch": (-2.6, 0.7),       # - = raise forward
             "shoulder_roll": (-0.15, 1.7),       # + = outward (left arm)
             "shoulder_yaw": (-0.9, 0.9),
             "elbow": (-0.3, 1.55)},              # 0 = forearm level, 1.57 = straight
)
