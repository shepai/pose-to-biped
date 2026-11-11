"""
Robot descriptions: everything the pipeline needs to know about one humanoid, in one object.

A RobotSpec lists the robot's driven joints and says which *slot* each one fills. Slots are a fixed
list of 31 joint roles (SLOTS) shared by every robot:

    0-11   legs   left then right: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    12-14  waist  yaw, roll, pitch
    15-28  arms   left then right: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                  wrist_roll, wrist_pitch, wrist_yaw
    29-30  head   yaw, pitch

A robot fills the slots it has joints for (the G1 fills 0-28, the H1 19 of them). Slots do two jobs:
  * they name joint roles, so code can ask for "the leg joints" or "the left arm" of any robot,
  * they give one policy a common observation and action layout for every robot
    (PoseTrackingEnv(layout="universal")): entry i always means the same joint role.

Adding a robot: write a file like h1.py with its joints, keypoint bodies and feet, register it in
robots/__init__.py, then run  python examples/RL/check_robot.py <name>  - it checks every name
against the model and prints what was measured from it (sole, standing height, size).

Anything left as None is measured from the MuJoCo model when it is loaded (see
G1_tele.motion.kinematics.Kinematics): PD gains and torque limits from the model's actuators, the
sole from the foot's collision geoms, the standing height and the size from the default pose.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"
MOTION_DIR = ASSETS / "motions"

SIDES = ("left", "right")
LEG_PARTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
WAIST_PARTS = ("waist_yaw", "waist_roll", "waist_pitch")
ARM_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HEAD_PARTS = ("head_yaw", "head_pitch")

SLOTS = tuple([f"{s}_{p}" for s in SIDES for p in LEG_PARTS] + list(WAIST_PARTS)
              + [f"{s}_{p}" for s in SIDES for p in ARM_PARTS] + list(HEAD_PARTS))
N_SLOTS = len(SLOTS)                                    # 31
SLOT_INDEX = {s: i for i, s in enumerate(SLOTS)}


def slot_part(slot):
    """'left_hip_pitch' -> 'hip_pitch'; 'waist_yaw' -> 'waist_yaw'."""
    for s in SIDES:
        if slot.startswith(s + "_"):
            return slot[len(s) + 1:]
    return slot


def slot_side(slot):
    for s in SIDES:
        if slot.startswith(s + "_"):
            return s
    return None


def slot_group(slot):
    p = slot_part(slot)
    return "leg" if p in LEG_PARTS else "waist" if p in WAIST_PARTS else "arm" if p in ARM_PARTS else "head"


def slot_axis(slot):
    """The axis a joint in this slot turns about when the robot stands in its default pose
    (x forward, y left, z up). Used to give every robot the same sign convention per slot."""
    p = slot_part(slot)
    if p.endswith("roll"):
        return np.array([1.0, 0.0, 0.0])
    if p.endswith("yaw"):
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])                    # pitch joints, knee, elbow


# How a left joint value becomes the mirrored right one: pitch-like joints keep their sign,
# roll and yaw joints flip (override per robot with RobotSpec.mirror).
def default_mirror(part):
    return -1.0 if (part.endswith("roll") or part.endswith("yaw")) else 1.0


# ---- per-slot numbers tuned on the G1; other robots reuse them by slot
#                      hip_p hip_r hip_y knee ank_p ank_r
_LEG_RES = [0.5, 0.4, 0.3, 0.5, 0.7, 0.3]               # ankles/hips: enough torque to move the
_LEG_ABS = [1.0, 0.6, 0.6, 1.2, 0.7, 0.3]               # centre of pressure across the whole sole
_ARM_RES = [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3]
_ARM_ABS = [1.6, 1.4, 1.2, 1.2, 0.5, 0.5, 0.5]
RESIDUAL_SCALE = np.array(_LEG_RES * 2 + [0.3, 0.2, 0.2] + _ARM_RES * 2 + [0.3, 0.3])     # rad per unit action
ABSOLUTE_SCALE = np.array(_LEG_ABS * 2 + [0.6, 0.4, 0.4] + _ARM_ABS * 2 + [0.5, 0.5])

# G1 measurements the motion generators' numbers were tuned for; other robots are scaled against them.
G1_LEG_LENGTH = 0.6234                                  # hip -> knee -> ankle
G1_HIP_HALF_WIDTH = 0.1165
G1_STAND_HEIGHT = 0.793                                 # base height the G1 numbers refer to ...
G1_STAND_CLEARANCE = 0.0088                             # ... at which its soles are this far above the floor


@dataclass(frozen=True)
class Joint:
    """One driven joint. name: the joint's name in the xml; slot: its role (one of SLOTS).
    q0: angle in the default (standing) pose. kp, kd, tau, armature: PD gains, torque limit and rotor
    inertia; None = take it from the model. locked: held at the reference pose, not driven by the
    policy (e.g. wrists, head). sign: +1/-1 to match the slot's sign convention; None = worked out
    from the joint's axis in the default pose."""
    name: str
    slot: str
    q0: float = 0.0
    kp: float = None
    kd: float = None
    tau: float = None
    armature: float = None
    locked: bool = False
    sign: float = None


def _arr(values, dtype=np.float32):
    return np.array([np.nan if v is None else v for v in values], dtype)


@dataclass
class RobotSpec:
    name: str
    xml_path: Path                                      # a scene with the robot AND a floor plane
    joints: tuple                                       # of Joint; sorted into slot order on creation
    keypoints: dict                                     # keypoint name -> (body name, offset in that body's frame)
    feet: tuple                                         # (left, right) foot body names (the bodies the soles are on)
    root_body: str = "pelvis"                           # the floating-base body
    pelvis_body: str = None                             # the body the legs hang from, if it is not the base
    # ---- feet; None = measured from the foot bodies' collision geoms
    ankle_height: float = None                          # foot-body origin above the floor, sole flat on it
    foot_center: tuple = None                           # centre of the sole in the foot frame (x, y)
    foot_half: tuple = None                             # half length, half width of the sole
    sole_points: tuple = None                           # contact points in the LEFT foot's frame ((x, y, z, r), ...);
    #                                                     the right foot's are mirrored in y. None = measured
    planted_height: float = None                        # a reference ankle below this is "on the floor"
    #                                                     (None = ankle_height + 0.015)
    # ---- size; None = measured in the default pose
    stand_height: float = None                          # base height in the default pose
    stand_clearance: float = 0.0                        # sole height above the floor at stand_height
    stance_x: float = None                              # ankle x relative to the base in the default pose
    scale: float = None                                 # leg length relative to the G1 (lengths, heights, speeds)
    lateral_scale: float = None                         # hip width relative to the G1 (stance width, side steps)
    gait_height: tuple = (0.70, 0.745)                  # walking base height range, in G1 terms (Kinematics.to_height):
    #                                                     raise it for robots whose ankles can't bend far (less knee bend)
    # ---- random upper-body poses for the synthetic motions
    explore: dict = field(default_factory=dict)         # part ("shoulder_pitch", "waist_yaw") -> (lo, hi) rad,
    #                                                     written for the LEFT side
    mirror: dict = field(default_factory=dict)          # part -> +1/-1 where it differs from default_mirror()
    motion_file: Path = None                            # default motion library (None = motions/<name>/synthetic.npz)
    loco_motion_file: Path = None                       # walking clips (None = motions/<name>/locomotion.npz)
    notes: str = ""

    def __post_init__(self):
        self.xml_path = Path(self.xml_path)
        js = list(self.joints)
        unknown = [j.slot for j in js if j.slot not in SLOT_INDEX]
        if unknown:
            raise ValueError(f"{self.name}: unknown slots {unknown}; valid: {SLOTS}")
        if len({j.slot for j in js}) != len(js):
            raise ValueError(f"{self.name}: two joints share a slot")
        js.sort(key=lambda j: SLOT_INDEX[j.slot])
        self.joints = tuple(js)
        if self.motion_file is None:
            self.motion_file = MOTION_DIR / self.name / "synthetic.npz"
        if self.loco_motion_file is None:
            self.loco_motion_file = MOTION_DIR / self.name / "locomotion.npz"
        missing = [k for k in ("left_hand", "right_hand", "left_foot", "right_foot", "left_elbow", "right_elbow",
                               "left_knee", "right_knee", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
                   if k not in self.keypoints]
        if missing:
            raise ValueError(f"{self.name}: keypoints missing {missing}")

    # ---------------------------------------------------------------- joints as arrays
    @property
    def n_joints(self):
        return len(self.joints)

    @property
    def joint_names(self):
        return [j.name for j in self.joints]

    @property
    def slots(self):
        return [j.slot for j in self.joints]

    @property
    def slot_idx(self):
        """Slot number (0..N_SLOTS-1) of every joint."""
        return np.array([SLOT_INDEX[j.slot] for j in self.joints])

    @property
    def default_q(self):
        return np.array([j.q0 for j in self.joints], np.float32)

    @property
    def default_q64(self):
        return np.array([j.q0 for j in self.joints], np.float64)

    @property
    def kp(self):
        return _arr([j.kp for j in self.joints])

    @property
    def kd(self):
        return _arr([j.kd for j in self.joints])

    @property
    def tau_lim(self):
        return _arr([j.tau for j in self.joints])

    @property
    def armature(self):
        return _arr([j.armature for j in self.joints])

    @property
    def signs(self):
        """Explicit slot signs (NaN where they are to be worked out from the model)."""
        return _arr([j.sign for j in self.joints], np.float64)

    @property
    def locked_idx(self):
        return [i for i, j in enumerate(self.joints) if j.locked]

    @property
    def ctrl_idx(self):
        """Joints the policy drives (everything that is not locked)."""
        return [i for i, j in enumerate(self.joints) if not j.locked]

    def group_idx(self, group, side=None):
        """Indices of the joints in a group: "leg", "waist", "arm" or "head" (optionally one side)."""
        return [i for i, j in enumerate(self.joints)
                if slot_group(j.slot) == group and (side is None or slot_side(j.slot) == side)]

    @property
    def leg_idx(self):
        return self.group_idx("leg")

    @property
    def upper_idx(self):
        """Everything above the legs (waist, arms, head)."""
        return [i for i, j in enumerate(self.joints) if slot_group(j.slot) != "leg"]

    def index(self, slot):
        """Index of the joint in a slot, or None if this robot has no such joint."""
        for i, j in enumerate(self.joints):
            if j.slot == slot:
                return i
        return None

    @property
    def residual_scale(self):
        return RESIDUAL_SCALE[self.slot_idx].copy()

    @property
    def absolute_scale(self):
        return ABSOLUTE_SCALE[self.slot_idx].copy()

    def mirror_sign(self, part):
        return float(self.mirror.get(part, default_mirror(part)))

    @property
    def has_ankle_roll(self):
        return self.index("left_ankle_roll") is not None

    def with_xml(self, xml_path):
        """A copy of this description that loads another model file (e.g. the G1 with hands)."""
        import copy
        s = copy.copy(self)
        s.xml_path = Path(xml_path)
        return s
