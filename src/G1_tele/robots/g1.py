"""
Unitree G1, 29 joints (unitree_rl_gym's g1_29dof.xml). These are the numbers the package was
written around; every other robot is described the same way and scaled against this one.

PD gains and torque limits: unitree_rl_gym. Rotor inertia: BeyondMimic's G1 configuration
(7520-14, 7520-22, 5020 (x2 for the parallel-linkage ankles / waist roll+pitch), 4010 for the wrists).
"""
from .spec import ARM_PARTS, ASSETS, LEG_PARTS, MOTION_DIR, SIDES, WAIST_PARTS, Joint, RobotSpec

_A14, _A22, _A50, _A40 = 0.010177520, 0.025101925, 0.003609725, 0.00425

#        part            q0    kp   kd  tau  armature
_LEG = [("hip_pitch",   -0.1, 100, 2,  88, _A14),
        ("hip_roll",     0.0, 100, 2, 139, _A22),
        ("hip_yaw",      0.0, 100, 2,  88, _A14),
        ("knee",         0.3, 150, 4, 139, _A22),
        ("ankle_pitch", -0.2,  40, 2,  50, 2 * _A50),
        ("ankle_roll",   0.0,  40, 2,  50, 2 * _A50)]
_WAIST = [("waist_yaw",   0.0, 300, 5, 88, _A14),
          ("waist_roll",  0.0, 300, 5, 50, 2 * _A50),
          ("waist_pitch", 0.0, 300, 5, 50, 2 * _A50)]
#        part              q0 (left)  kp  kd  tau armature  locked
_ARM = [("shoulder_pitch", 0.0,  60, 2, 25, _A50, False),
        ("shoulder_roll",  0.15, 60, 2, 25, _A50, False),      # right: -0.15
        ("shoulder_yaw",   0.0,  60, 2, 25, _A50, False),
        ("elbow",          0.3,  60, 2, 25, _A50, False),
        ("wrist_roll",     0.0,  30, 1, 25, _A50, True),       # wrists are held still (no keypoint drives them)
        ("wrist_pitch",    0.0,  30, 1,  5, _A40, True),
        ("wrist_yaw",      0.0,  30, 1,  5, _A40, True)]
assert [p for p, *_ in _LEG] == list(LEG_PARTS) and [p for p, *_ in _WAIST] == list(WAIST_PARTS) \
    and [p for p, *_ in _ARM] == list(ARM_PARTS)

_joints = []
for side in SIDES:
    for part, q0, kp, kd, tau, arm in _LEG:
        _joints.append(Joint(f"{side}_{part}_joint", f"{side}_{part}", q0, kp, kd, tau, arm))
for part, q0, kp, kd, tau, arm in _WAIST:
    _joints.append(Joint(f"{part}_joint", part, q0, kp, kd, tau, arm))
for side in SIDES:
    for part, q0, kp, kd, tau, arm, locked in _ARM:
        if part == "shoulder_roll" and side == "right":
            q0 = -q0
        _joints.append(Joint(f"{side}_{part}_joint", f"{side}_{part}", q0, kp, kd, tau, arm, locked))

KEYPOINTS = {
    "left_hand": ("left_wrist_yaw_link", (0.08, 0.0, 0.0)),      # palm centre
    "right_hand": ("right_wrist_yaw_link", (0.08, 0.0, 0.0)),
    "left_elbow": ("left_elbow_link", (0.0, 0.0, 0.0)),
    "right_elbow": ("right_elbow_link", (0.0, 0.0, 0.0)),
    "left_shoulder": ("left_shoulder_roll_link", (0.0, 0.0, 0.0)),
    "right_shoulder": ("right_shoulder_roll_link", (0.0, 0.0, 0.0)),
    "left_hip": ("left_hip_roll_link", (0.0, 0.0, 0.0)),
    "right_hip": ("right_hip_roll_link", (0.0, 0.0, 0.0)),
    "left_knee": ("left_knee_link", (0.0, 0.0, 0.0)),
    "right_knee": ("right_knee_link", (0.0, 0.0, 0.0)),
    "left_foot": ("left_ankle_roll_link", (0.0, 0.0, 0.0)),       # ankle
    "right_foot": ("right_ankle_roll_link", (0.0, 0.0, 0.0)),
    "left_toe": ("left_ankle_roll_link", (0.12, 0.0, -0.03)),
    "right_toe": ("right_ankle_roll_link", (0.12, 0.0, -0.03)),
    "head": ("torso_link", (0.0, 0.0, 0.43)),
}

G1 = RobotSpec(
    name="g1",
    xml_path=ASSETS / "g1_description" / "g1_29dof.xml",
    joints=tuple(_joints),
    keypoints=KEYPOINTS,
    feet=("left_ankle_roll_link", "right_ankle_roll_link"),
    root_body="pelvis",
    ankle_height=0.035,               # ankle-frame origin above the floor when the sole is flat on it
    foot_center=(0.035, 0.0),         # centre of the sole in the ankle frame (x, y)
    foot_half=(0.085, 0.03),          # half length / half width of the sole
    planted_height=0.05,
    stand_height=0.793, stand_clearance=0.0088, stance_x=-0.026, scale=1.0, lateral_scale=1.0,
    explore={"waist_yaw": (-0.6, 0.6), "waist_roll": (-0.2, 0.2), "waist_pitch": (-0.1, 0.35),   # + = lean forward
             "shoulder_pitch": (-2.6, 0.7),       # - = raise forward
             "shoulder_roll": (-0.15, 1.7),       # + = outward (left arm)
             "shoulder_yaw": (-0.9, 0.9),
             "elbow": (-0.7, 1.55)},              # 1.57 = straight
    motion_file=MOTION_DIR / "synthetic.npz",
    loco_motion_file=MOTION_DIR / "locomotion.npz",
)
