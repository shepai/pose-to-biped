"""
Booster T1, 23 joints (MuJoCo Menagerie's booster_t1/scene.xml; fetch it with examples/get_robots.py).

Differences from the G1 that matter here:
  * the floating base is the Trunk (the chest), not the pelvis: the legs hang from the "Waist" body,
    which turns under the trunk. pelvis_body tells the motion tools which body is the pelvis, and the
    waist joint's sign is flipped so that "waist yaw" still means the chest turning over the hips.
  * the arms point sideways at zero (a T-pose): shoulder roll -1.4 brings the left arm down. The
    joint the model calls Elbow_Pitch twists the upper arm (the G1's shoulder yaw), and Elbow_Yaw
    bends the elbow (0 = straight, negative = bent for the left arm).
  * the head joints are held still (no keypoint drives them).

Leg gains: booster_gym (envs/T1.yaml: hip 200/5, knee 200/5, ankle 50/1); the waist uses the hip's.
Arm and head gains and all torque limits are read from the model's position actuators (kp 75, kv 5).
The default pose is Menagerie's "home" keyframe.
"""
from .spec import ASSETS, Joint, RobotSpec

_joints = [
    Joint("AAHead_yaw", "head_yaw", locked=True),
    Joint("Head_pitch", "head_pitch", locked=True),
    Joint("Waist", "waist_yaw", 0.0, 200, 5, sign=-1.0),
]
for Side, side, sgn in (("Left", "left", 1.0), ("Right", "right", -1.0)):
    _joints += [
        Joint(f"{Side}_Shoulder_Pitch", f"{side}_shoulder_pitch", 0.0),
        Joint(f"{Side}_Shoulder_Roll", f"{side}_shoulder_roll", -1.4 * sgn),
        Joint(f"{Side}_Elbow_Pitch", f"{side}_shoulder_yaw", 0.0),            # twists the upper arm
        Joint(f"{Side}_Elbow_Yaw", f"{side}_elbow", -0.4 * sgn),              # bends the elbow
        Joint(f"{Side}_Hip_Pitch", f"{side}_hip_pitch", -0.2, 200, 5),
        Joint(f"{Side}_Hip_Roll", f"{side}_hip_roll", 0.0, 200, 5),
        Joint(f"{Side}_Hip_Yaw", f"{side}_hip_yaw", 0.0, 200, 5),
        Joint(f"{Side}_Knee_Pitch", f"{side}_knee", 0.4, 200, 5),
        Joint(f"{Side}_Ankle_Pitch", f"{side}_ankle_pitch", -0.2, 50, 1),
        Joint(f"{Side}_Ankle_Roll", f"{side}_ankle_roll", 0.0, 50, 1),
    ]

_kp = {}
for Side, side, sgn, arm in (("Left", "left", 1.0, "AL"), ("Right", "right", -1.0, "AR")):
    _kp.update({
        f"{side}_hand": (f"{side}_hand_link", (0.0, 0.19 * sgn, 0.0)),        # the arm runs along the body's y axis
        f"{side}_elbow": (f"{side}_hand_link", (0.0, 0.0, 0.0)),
        f"{side}_shoulder": (f"{arm}2", (0.0, 0.0, 0.0)),
        f"{side}_hip": (f"Hip_Roll_{Side}", (0.0, 0.0, 0.0)),
        f"{side}_knee": (f"Shank_{Side}", (0.0, 0.0, 0.0)),
        f"{side}_foot": (f"{side}_foot_link", (0.0, 0.0, 0.0)),               # ankle
        f"{side}_toe": (f"{side}_foot_link", (0.1215, 0.0, -0.03)),
    })
_kp["head"] = ("H2", (0.01, 0.0, 0.11))

T1 = RobotSpec(
    name="t1",
    xml_path=ASSETS / "robots" / "booster_t1" / "scene.xml",
    joints=tuple(_joints),
    keypoints=_kp,
    feet=("left_foot_link", "right_foot_link"),
    root_body="Trunk",
    pelvis_body="Waist",
    explore={"waist_yaw": (-0.6, 0.6),
             "shoulder_pitch": (-2.6, 0.7),       # - = raise forward
             "shoulder_roll": (-1.7, 0.1),        # -1.57 = arm straight down, 0 = level (left arm)
             "shoulder_yaw": (-0.9, 0.9),
             "elbow": (-2.2, -0.05)},             # 0 = straight, -1.57 = right angle (left arm)
    mirror={"shoulder_yaw": 1.0, "elbow": -1.0},  # Elbow_Pitch turns about y, Elbow_Yaw about z
)
