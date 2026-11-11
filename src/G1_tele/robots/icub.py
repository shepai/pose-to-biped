"""
iCub 2.7 (IIT), 31 joints (23 driven by the policy; wrists and head held) + neck roll (held at 0). The model is converted from the
iCub team's URDF (robotology/icub-models, iCubGazeboV2_7; CC BY-SA 4.0) by examples/get_robots.py,
which turns the robot to face +x and adds a floor (G1_tele.robots.urdf_to_mjcf).

About 1.05 m tall and 33 kg, so close to the G1 in size; the legs are a little shorter (0.84 x).
Differences from the G1 that matter here:
  * every joint uses the same sign convention on both sides (a mirrored pose has the SAME angles
    on the left and right), so `mirror` is +1 for everything,
  * the shoulders are tilted 15 degrees, the elbows can't straighten fully (0.26 rad at most) and
    the shoulder roll only goes outward (0 = arm against the body),
  * a tall ankle (7.5 cm from the ankle axis down to the sole) and a 16 x 7 cm sole plate,
  * the ankles only bend +-0.61 rad (the G1's 0.87 forward), so iCub walks with straighter legs than
    the G1 (gait_height): at the G1's knee bend the back foot would run out of ankle and tip up.

Numbers that are this package's choices, not the iCub team's (check them with
examples/RL/check_robot.py icub --stand before training):
  * PD gains: the G1's, scaled to iCub's smaller torques,
  * torque limits: legs from the URDF (55.5 Nm hip pitch, 37 Nm elsewhere); the URDF has
    placeholders for the rest, so torso and shoulders use iCub's high-power motor group (40 Nm),
    the elbow the medium group (20 Nm), the wrists the low-power group (11 Nm, 5 Nm for the
    tendon-driven pitch / yaw) and the neck its 8 Nm (Parmiggiani et al., "The design of the iCub
    humanoid robot", IJHR 2012),
  * rotor inertia (armature, written into the converted model by get_robots.py: the URDF has none),
  * the default pose (knees bent 0.6 rad, arms slightly out and forward).
"""
from .spec import ASSETS, SIDES, Joint, RobotSpec

#        xml name     slot part       q0 (left)  kp   kd   tau   locked
_LEG = [("hip_pitch",  "hip_pitch",    0.30,  100, 2.0, 55.5, False),     # + = thigh forward
        ("hip_roll",   "hip_roll",     0.0,   100, 2.0, 37.0, False),     # + = outward (both sides)
        ("hip_yaw",    "hip_yaw",      0.0,   100, 2.0, 37.0, False),
        ("knee",       "knee",        -0.60,  120, 3.0, 37.0, False),     # - = bend
        ("ankle_pitch", "ankle_pitch", -0.30,  40, 1.5, 37.0, False),
        ("ankle_roll", "ankle_roll",   0.0,    40, 1.5, 37.0, False)]
_ARM = [("shoulder_pitch", "shoulder_pitch", -0.15, 40, 1.0, 40.0, False),   # - = raise forward
        ("shoulder_roll",  "shoulder_roll",   0.30, 40, 1.0, 40.0, False),   # + = outward (both sides)
        ("shoulder_yaw",   "shoulder_yaw",    0.0,  40, 1.0, 40.0, False),
        ("elbow",          "elbow",           0.50, 30, 1.0, 20.0, False),   # 0.26 = nearly straight
        ("wrist_prosup",   "wrist_roll",      0.0,  10, 0.3, 11.0, True),
        ("wrist_pitch",    "wrist_pitch",     0.0,  10, 0.3,  5.0, True),
        ("wrist_yaw",      "wrist_yaw",       0.0,  10, 0.3,  5.0, True)]

_joints = [Joint("torso_yaw", "waist_yaw", 0.0, 150, 3.0, 40.0),
           Joint("torso_roll", "waist_roll", 0.0, 150, 3.0, 40.0),
           Joint("torso_pitch", "waist_pitch", 0.0, 150, 3.0, 40.0),
           Joint("neck_yaw", "head_yaw", 0.0, 10, 0.3, 8.0, locked=True),
           Joint("neck_pitch", "head_pitch", 0.0, 10, 0.3, 8.0, locked=True)]
for side in SIDES:
    s = side[0]
    for name, part, q0, kp, kd, tau, locked in _LEG + _ARM:
        _joints.append(Joint(f"{s}_{name}", f"{side}_{part}", q0, kp, kd, tau, locked=locked))

# offsets measured on the converted model (zero pose): shoulder = where the pitch and roll axes
# cross, elbow = on the upper arm's centre line, hand = centre of the hand, toe = front of the sole
_kp = {"head": ("head", (-0.0077, 0.1184, -0.0165))}
for side, s, sy in (("left", "l", 1.0), ("right", "r", -1.0)):
    _kp.update({
        f"{side}_hand": (f"{s}_hand", (0.0631 * sy, 0.0076, 0.0471)),
        f"{side}_elbow": (f"{s}_elbow_1", (0.0, 0.034 if s == "l" else 0.0225, 0.0)),
        f"{side}_shoulder": (f"{s}_shoulder_2", (0.003 * sy, 0.0, -0.0114)),
        f"{side}_hip": (f"{s}_hip_2", (0.0, 0.0, 0.0)),
        f"{side}_knee": (f"{s}_lower_leg", (0.0, 0.0, 0.0)),
        f"{side}_foot": (f"{s}_ankle_2", (0.0, 0.0, 0.0)),                   # ankle
        f"{side}_toe": (f"{s}_ankle_2", (0.1065, 0.005 * sy, -0.075)),
    })

ICUB = RobotSpec(
    name="icub",
    xml_path=ASSETS / "robots" / "icub" / "scene.xml",
    joints=tuple(_joints),
    keypoints=_kp,
    feet=("l_ankle_2", "r_ankle_2"),
    root_body="root_link",
    explore={"waist_yaw": (-0.6, 0.6),
             "waist_roll": (-0.2, 0.2),
             "waist_pitch": (-0.1, 0.35),          # + = lean forward
             "shoulder_pitch": (-1.6, 0.15),       # - = raise forward
             "shoulder_roll": (0.05, 1.7),         # + = outward
             "shoulder_yaw": (-0.6, 0.9),
             "elbow": (0.3, 1.8)},                 # 0.26 = nearly straight
    gait_height=(0.75, 0.775),              # walk with straighter legs: the ankles only bend 0.61 rad
    mirror={p: 1.0 for p in ("hip_roll", "hip_yaw", "ankle_roll", "shoulder_roll", "shoulder_yaw",
                             "wrist_roll", "wrist_yaw", "waist_roll", "waist_yaw", "head_yaw")},
    notes="Converted from URDF (icub-models, CC BY-SA 4.0). Gains, arm torque limits and rotor inertia "
          "are this package's choices.",
)

# rotor inertia written into the converted model (kg m^2, at the joint), by joint name suffix
ARMATURE = {"hip_pitch": 0.05, "hip_roll": 0.05, "hip_yaw": 0.03, "knee": 0.05, "ankle_pitch": 0.03,
            "ankle_roll": 0.03, "torso_pitch": 0.05, "torso_roll": 0.05, "torso_yaw": 0.03,
            "shoulder_pitch": 0.03, "shoulder_roll": 0.03, "shoulder_yaw": 0.02, "elbow": 0.02,
            "wrist_prosup": 0.01, "wrist_pitch": 0.005, "wrist_yaw": 0.005,
            "neck_pitch": 0.01, "neck_roll": 0.01, "neck_yaw": 0.01}


def armature_for(joint_name):
    for suffix, a in ARMATURE.items():
        if joint_name.endswith(suffix):
            return a
    return None
