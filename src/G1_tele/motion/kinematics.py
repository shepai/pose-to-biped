"""
Keypoints, forward kinematics and a whole-body IK solver for any robot in G1_tele.robots
(no torch, plain MuJoCo + numpy).

    k = Kinematics("h1")          # or Kinematics() / G1Kinematics() for the G1

Keypoint frame ("heading frame")
--------------------------------
All keypoint targets the env and the retargeting tools exchange are expressed in the robot's
heading frame:  origin = base (x, y) projected onto the ground (z = 0 is the floor),
axes = base yaw only (x forward, y left, z up).  So a keypoint's z is its height above the
floor, and x/y are relative to the base.  This is also the natural frame for a body tracker:
take each joint relative to the hips' mid-point, rotate by the body's heading, add the pelvis
height, scale by robot_size / human_size.

Measured robot geometry
-----------------------
Kinematics measures what the motion tools and the env need from the model, in the default pose,
unless the robot's description gives it (see G1_tele.robots.spec):
  * the sole of each foot: its plane, contact points (sole_points, for contact detection and the
    support polygon), centre and half size, the foot body's orientation when the sole is flat
    (foot_flat), and ankle_height (foot-body origin above the floor when the sole is flat on it),
  * standing height, stance (where the feet are relative to the base), leg length, hip width,
  * scale / lateral_scale: leg length and hip width relative to the G1. Motion generators were
    tuned on the G1 and convert their numbers with to_height(), to_length() and to_half_width().
"""
from pathlib import Path

import numpy as np
import mujoco

from G1_tele.robots import get_robot
from G1_tele.robots.spec import G1_HIP_HALF_WIDTH, G1_LEG_LENGTH, SIDES

# ---- G1 values, kept for code written for the G1 only (other robots: use a Kinematics instance)
# name -> (body, offset in that body's frame)
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
KEYPOINT_SETS = {
    "minimal": ("left_hand", "right_hand", "left_foot", "right_foot"),
    "limbs": ("left_hand", "right_hand", "left_foot", "right_foot",
              "left_elbow", "right_elbow", "left_knee", "right_knee"),
    "full": ("left_hand", "right_hand", "left_foot", "right_foot",
             "left_elbow", "right_elbow", "left_knee", "right_knee",
             "left_shoulder", "right_shoulder", "left_hip", "right_hip"),
}
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")      # G1
ANKLE_HEIGHT = 0.035          # ankle-frame origin above the floor when the sole is flat on it
FOOT_CENTER = np.array([0.035, 0.0])   # centre of the sole in the ankle frame (x, y)
FOOT_HALF = np.array([0.085, 0.03])    # half length / half width of the sole


# ---------------------------------------------------------------- small rotation helpers (wxyz)
def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of_quat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec_from_mat(R):
    """Axis-angle vector of a rotation matrix (log map)."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    ang = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * v
    if np.pi - ang < 1e-4:                         # near 180 deg: use the diagonal
        axis = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        axis *= np.sign(v + 1e-12)
        return ang * axis / max(np.linalg.norm(axis), 1e-9)
    return ang / (2 * np.sin(ang)) * v


def slerp(q0, q1, f):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)


# ---------------------------------------------------------------- measured geometry
CONTACT_GAP = 0.005           # a sole point whose lowest point is closer than this to the floor touches it
_GEOMETRY = {}                # (robot name, xml path) -> measurements (the model is the same for every env)


def _geom_points(m, d, g):
    """Points (n, 3, world) and radii (n,) whose 'lowest point' hull is geom g's underside."""
    t = int(m.geom_type[g])
    p, R, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    T = mujoco.mjtGeom
    if t == T.mjGEOM_SPHERE:
        return p[None], np.array([s[0]])
    if t == T.mjGEOM_CAPSULE:
        return np.stack([p - R[:, 2] * s[1], p + R[:, 2] * s[1]]), np.full(2, s[0])
    if t in (T.mjGEOM_BOX, T.mjGEOM_ELLIPSOID):
        c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * s[:3]
        return p + c @ R.T, np.zeros(8)
    if t == T.mjGEOM_CYLINDER:
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.stack([s[0] * np.cos(a), s[0] * np.sin(a), np.zeros_like(a)], 1)
        c = np.concatenate([ring + [0, 0, -s[1]], ring + [0, 0, s[1]]])
        return p + c @ R.T, np.zeros(len(c))
    if t == T.mjGEOM_MESH:
        i = int(m.geom_dataid[g])
        v = np.asarray(m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]], float)
        return p + v @ R.T, np.zeros(len(v))
    raise ValueError(f"cannot measure a sole from geom type {t}")


def _hull2(pts):
    """Indices of the 2-D convex hull of pts (n, 2), counter-clockwise."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    if len(order) <= 2:
        return order

    def half(seq):
        h = []
        for i in seq:
            while len(h) >= 2 and ((pts[h[-1]][0] - pts[h[-2]][0]) * (pts[i][1] - pts[h[-2]][1])
                                   - (pts[h[-1]][1] - pts[h[-2]][1]) * (pts[i][0] - pts[h[-2]][0])) <= 1e-12:
                h.pop()
            h.append(i)
        return h
    lo, hi = half(order), half(reversed(order))
    return lo[:-1] + hi[:-1]


def _measure_sole(m, d, body):
    """Sole of one foot in the current pose: (R_sole (3,3) world, contact points (n,3) world,
    sphere centres + radii (n,4) world) - the sole plane is fitted to the lowest collision points."""
    pts, rad = [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g]):
            p, r = _geom_points(m, d, g)
            pts.append(p); rad.append(r)
    if not pts:
        raise ValueError(f"body {m.body(body).name if hasattr(m.body(body), 'name') else body} has no collision geoms")
    P, r = np.concatenate(pts), np.concatenate(rad)
    n = np.array([0.0, 0.0, 1.0])
    for band in (0.01, 0.003, 0.002):                     # fit the plane to the lowest points, tighten, refit
        c = P - r[:, None] * n
        h = c @ n
        sel = h < h.min() + band
        cs = c[sel]
        if len(cs) >= 3:
            u, sv, vt = np.linalg.svd(cs - cs.mean(0))
            if sv[1] > 1e-3 * max(sv[0], 1e-9):           # not collinear: a plane is defined
                nn = vt[2] * np.sign(vt[2][2])
                if nn[2] > np.cos(np.radians(20)):
                    n = nn
    c = P - r[:, None] * n
    h = c @ n
    sel = h < h.min() + 0.002
    fwd = np.array([1.0, 0.0, 0.0]) - n[0] * n            # the default pose faces +x
    fwd /= np.linalg.norm(fwd)
    R_sole = np.stack([fwd, np.cross(n, fwd), n], 1)
    keep = np.flatnonzero(sel)
    local = (c[keep] - c[keep].mean(0)) @ R_sole
    if len(keep) > 3:                                     # many mesh vertices: keep the hull corners
        hull = _hull2(local[:, :2])
        if len(hull) >= 3:
            keep = keep[hull]
            if len(keep) > 12:
                keep = keep[np.linspace(0, len(keep) - 1, 12).astype(int)]
    return R_sole, c[keep], np.concatenate([P[keep], r[keep, None]], 1)


def measure_geometry(spec, m):
    """Everything Kinematics and the env need to know about the robot's size and feet (see the module
    docstring). Values given in the RobotSpec win over measured ones."""
    key = (spec.name, str(Path(spec.xml_path).resolve()))
    if key in _GEOMETRY:
        return _GEOMETRY[key]
    d = mujoco.MjData(m)
    q_adr = np.array([m.joint(n).qposadr[0] for n in spec.joint_names])
    root = m.body(spec.root_body).id
    q = m.qpos0.copy()
    q[:7] = [0, 0, 2.0, 1, 0, 0, 0]
    q[q_adr] = spec.default_q64
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    g = {}
    feet = [m.body(n).id for n in spec.feet]
    flat, sole_pts, low = [], [], []
    for b in feet:
        R_sole, c, spheres = _measure_sole(m, d, b)
        Rf, pf = d.xmat[b].reshape(3, 3), d.xpos[b]
        flat.append(Rf.T @ R_sole)                        # sole axes in the foot frame
        sp = spheres.copy()
        sp[:, :3] = (spheres[:, :3] - pf) @ Rf            # sole points in the foot frame
        sole_pts.append(sp)
        low.append(((c - pf) @ R_sole))                   # contact points in the sole frame, foot origin = 0
    both = np.concatenate(low)
    ankle = float(-np.mean(both[:, 2]))
    xs = [lp[:, 0] for lp in low]
    ys = [lp[:, 1] for lp in low]
    cx = float(np.mean([(x.max() + x.min()) / 2 for x in xs]))
    hx = float(np.mean([(x.max() - x.min()) / 2 for x in xs]))
    hy = float(np.mean([(y.max() - y.min()) / 2 for y in ys]))
    cy = float(np.mean([s * (y.max() + y.min()) / 2 for s, y in zip((1, -1), ys)]))   # + = outward
    g["ankle_height"] = spec.ankle_height if spec.ankle_height is not None else ankle
    g["foot_center"] = np.array(spec.foot_center if spec.foot_center is not None else (cx, 0.0), float)
    g["foot_center_lateral"] = cy                         # how far the sole centre sits outward of the ankle
    g["foot_half"] = np.array(spec.foot_half if spec.foot_half is not None else (hx, hy), float)
    g["foot_flat"] = flat
    if spec.sole_points is not None:
        left = np.array(spec.sole_points, float)
        g["sole_points"] = [left, left * np.array([1, -1, 1, 1])]
    else:
        g["sole_points"] = sole_pts
    g["planted_height"] = spec.planted_height if spec.planted_height is not None else g["ankle_height"] + 0.015

    # standing: lowest sole point on the floor (+ clearance)
    lowest = np.inf
    for sp, b in zip(g["sole_points"], feet):
        z = (sp[:, :3] @ d.xmat[b].reshape(3, 3).T + d.xpos[b])[:, 2] - sp[:, 3]
        lowest = min(lowest, float(z.min()))
    stand = 2.0 - lowest + spec.stand_clearance
    g["stand_height"] = spec.stand_height if spec.stand_height is not None else stand
    g["stand_clearance"] = spec.stand_clearance
    shift = g["stand_height"] - 2.0

    def kp(name):
        body, off = spec.keypoints[name]
        b = m.body(body).id
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array(off, float) + np.array([0, 0, shift])
    P = {n: kp(n) for n in spec.keypoints}
    mean = lambda f: float(np.mean([f(s) for s in SIDES]))                       # noqa: E731
    dist = lambda a, b: mean(lambda s: np.linalg.norm(P[f"{s}_{a}"] - P[f"{s}_{b}"]))  # noqa: E731
    g["leg_length"] = dist("hip", "knee") + dist("knee", "foot")
    g["arm_length"] = dist("shoulder", "elbow") + dist("elbow", "hand")
    g["hip_half_width"] = mean(lambda s: abs(P[f"{s}_hip"][1]))
    g["stance_y"] = mean(lambda s: abs(P[f"{s}_foot"][1]))
    g["stance_x"] = spec.stance_x if spec.stance_x is not None else mean(lambda s: P[f"{s}_foot"][0])
    g["hip_drop"] = g["stand_height"] - mean(lambda s: P[f"{s}_hip"][2])          # base above the hips
    g["scale"] = spec.scale if spec.scale is not None else g["leg_length"] / G1_LEG_LENGTH
    g["lateral_scale"] = spec.lateral_scale if spec.lateral_scale is not None else g["hip_half_width"] / G1_HIP_HALF_WIDTH
    g["mass"] = float(m.body_subtreemass[root])
    g["head_height"] = float(P["head"][2]) if "head" in P else None
    _GEOMETRY[key] = g
    return g


def _g1_geometry():
    from G1_tele.robots import ROBOTS
    spec = ROBOTS["g1"]
    key = (spec.name, str(Path(spec.xml_path).resolve()))
    if key not in _GEOMETRY:
        measure_geometry(spec, mujoco.MjModel.from_xml_path(str(spec.xml_path)))
    return _GEOMETRY[key]


# ---------------------------------------------------------------- kinematic model
class Kinematics:
    """A private MjModel/MjData used purely for kinematics (FK, Jacobians, IK) - never stepped.

    robot: a name ("g1", "h1", ...) or a RobotSpec; xml_path: load another model file for it."""

    def __init__(self, robot=None, xml_path=None):
        spec = get_robot(robot)
        if xml_path is not None and Path(xml_path) != Path(spec.xml_path):
            spec = spec.with_xml(xml_path)
        self.spec = spec
        if not Path(spec.xml_path).exists():
            raise FileNotFoundError(f"{spec.xml_path} not found - for robots other than the G1 run "
                                    "python examples/get_robots.py first")
        self.m = mujoco.MjModel.from_xml_path(str(spec.xml_path))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.nq, self.nv = m.nq, m.nv
        names = spec.joint_names
        self.joint_names = names
        self.n_joints = len(names)
        self.q_adr = np.array([m.joint(n).qposadr[0] for n in names])
        self.v_adr = np.array([m.joint(n).dofadr[0] for n in names])
        self.q_lo = np.array([m.joint(n).range[0] for n in names])
        self.q_hi = np.array([m.joint(n).range[1] for n in names])
        self.root_body = m.body(spec.root_body).id
        self.pelvis_id = m.body(spec.pelvis_body or spec.root_body).id
        self.foot_ids = [m.body(n).id for n in spec.feet]
        self.keypoints = dict(spec.keypoints)
        self.default_q = spec.default_q
        self.leg_idx, self.upper_idx = spec.leg_idx, spec.upper_idx
        self.locked_idx = spec.locked_idx
        g = measure_geometry(spec, m)
        self.geometry = g
        self.ankle_height, self.foot_center, self.foot_half = g["ankle_height"], g["foot_center"], g["foot_half"]
        self.sole_points, self.planted_height = g["sole_points"], g["planted_height"]
        self.stand_height, self.stance_x, self.stance_y = g["stand_height"], g["stance_x"], g["stance_y"]
        self.scale, self.lateral_scale = g["scale"], g["lateral_scale"]
        self.leg_length, self.hip_half_width, self.mass = g["leg_length"], g["hip_half_width"], g["mass"]
        self._foot_flat = g["foot_flat"]
        self.has_ankle_roll = spec.has_ankle_roll
        self._kp_cache = {}
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))

    # ---------- converting the motion generators' G1 numbers to this robot
    def to_length(self, x):
        """A G1 length (step length, lift height, sway) at this robot's size."""
        return x * self.scale

    def to_height(self, h):
        """A G1 base height (0.793 = standing) -> the base height that bends this robot's legs as much."""
        g1 = _g1_geometry()
        return self.geometry["hip_drop"] + self.ankle_height + (h - g1["hip_drop"] - g1["ankle_height"]) * self.scale

    def to_half_width(self, y):
        """A G1 stance half width (ankle y) -> this robot's, never so narrow that the feet touch."""
        g1 = _g1_geometry()
        out = self.stance_y + (y - g1["stance_y"]) * self.lateral_scale
        return max(out, self.min_half_width)

    @property
    def min_half_width(self):
        """Smallest ankle y that keeps 3 cm (x scale) between the soles: a sole's inner edge sits at
        ankle y + (outward offset of the sole centre) - half width."""
        return float(self.foot_half[1] - self.geometry["foot_center_lateral"] + 0.015 * self.scale)

    def foot_flat(self, side, yaw=0.0):
        """World orientation of foot `side` (0 = left, 1 = right, or "left"/"right"/"L"/"R") when its
        sole is flat on the floor and it points along yaw."""
        i = side if isinstance(side, (int, np.integer)) else (0 if str(side)[0].lower() == "l" else 1)
        return rot_z(yaw) @ self._foot_flat[i].T

    def foot_yaw(self, side, R_foot):
        """Heading of a foot from its body orientation (inverse of foot_flat)."""
        i = side if isinstance(side, (int, np.integer)) else (0 if str(side)[0].lower() == "l" else 1)
        S = R_foot @ self._foot_flat[i]
        return float(np.arctan2(S[1, 0], S[0, 0]))

    def sole_world(self, side, data=None):
        """Sole points of a foot in the world: (n, 3) centres and (n,) radii."""
        d = self.d if data is None else data
        b = self.foot_ids[side]
        sp = self.sole_points[side]
        return d.xpos[b] + sp[:, :3] @ d.xmat[b].reshape(3, 3).T, sp[:, 3]

    def joint_signs(self):
        """+1 / -1 per joint so that sign * angle turns the same way on every robot: about the slot's
        axis (roll x, pitch y, yaw z; see robots.spec.slot_axis) in the default pose. RobotSpec signs
        win; otherwise the joint axis in the default standing pose decides."""
        if not hasattr(self, "_signs"):
            from G1_tele.robots.spec import slot_axis
            self.forward(self.standing_qpos())
            out = self.spec.signs.copy()
            for i, jnt in enumerate(self.spec.joints):
                if np.isfinite(out[i]):
                    continue
                axis = np.asarray(self.d.xaxis[self.m.joint(jnt.name).id], float)
                dot = float(axis @ slot_axis(jnt.slot))
                out[i] = 1.0 if dot >= 0 else -1.0
            self._signs = out
        return self._signs

    # ---------- state <-> arrays
    def qpos_from(self, root, joints):
        q = self.m.qpos0.copy()
        q[:7] = root
        q[self.q_adr] = joints
        return q

    def default_qpos(self, height=None):
        h = self.stand_height if height is None else height
        return self.qpos_from(np.array([0, 0, h, 1, 0, 0, 0.0]), self.default_q)

    def standing_qpos(self):
        """Default pose with the soles exactly on the floor (stand_height minus any clearance)."""
        return self.default_qpos(self.stand_height - self.geometry["stand_clearance"])

    def keypoint_spec(self, names):
        key = tuple(names)
        if key not in self._kp_cache:
            missing = [n for n in names if n not in self.keypoints]
            if missing:
                raise KeyError(f"{self.spec.name} has no keypoints {missing}")
            self._kp_cache[key] = (np.array([self.m.body(self.keypoints[n][0]).id for n in names]),
                                   np.array([self.keypoints[n][1] for n in names], float))
        return self._kp_cache[key]

    # ---------- forward kinematics
    def forward(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)

    def keypoints_world(self, names, data=None):
        d = self.d if data is None else data
        ids, off = self.keypoint_spec(names)
        R = d.xmat[ids].reshape(-1, 3, 3)
        return d.xpos[ids] + np.einsum("kij,kj->ki", R, off)

    @staticmethod
    def heading(root_pos, root_quat):
        """(origin, R_yaw) of the heading frame for a root pose."""
        return np.array([root_pos[0], root_pos[1], 0.0]), rot_z(yaw_of_quat(np.asarray(root_quat)))

    def keypoints_local(self, names, qpos=None, data=None):
        """Keypoints in the heading frame (see module docstring)."""
        if qpos is not None:
            self.forward(qpos)
        d = self.d if data is None else data
        o, R = self.heading(d.qpos[:3], d.qpos[3:7])
        return (self.keypoints_world(names, d) - o) @ R

    # ---------- inverse kinematics
    def solve_ik(self, qpos, tasks, active=None, posture=None, posture_w=1e-3, iters=30, damping=1e-3,
                 tol=1e-4, max_step=0.25):
        """
        Damped least squares IK on the full floating-base model.

        tasks  : list of dicts, each one of
                 {"type": "point",  "body": id, "offset": (3,), "target": (3,) world, "w": weight}
                 {"type": "orient", "body": id, "target": (3,3) world rotation, "w": weight}
                 {"type": "com",    "target": (2,) or (3,) world, "w": weight}   (xy or xyz)
                 {"type": "root_height", "target": z, "w": weight}
                 {"type": "root_orient", "target": (3,3) world rotation, "w": weight}
        active : boolean mask over the nv dofs that IK may move (default: all)
        posture: target qpos the free dofs are gently pulled towards (default: current qpos)
        Returns (qpos, weighted residual norm).
        """
        m, d = self.m, self.d
        q = np.array(qpos, float)
        nv = self.nv
        act = np.ones(nv, bool) if active is None else np.asarray(active, bool)
        ia = np.flatnonzero(act)
        post = q.copy() if posture is None else np.asarray(posture, float)
        pw = np.full(nv, posture_w) if np.isscalar(posture_w) else np.asarray(posture_w, float)
        dq_post = np.zeros(nv)
        jp, jr = self._jacp, self._jacr
        err = np.inf
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            J_rows, e_rows, w_rows = [], [], []
            for t in tasks:
                ty, w = t["type"], t["w"]
                if ty == "point":
                    b = t["body"]
                    p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ t["offset"]
                    mujoco.mj_jac(m, d, jp, None, p, b)
                    J_rows.append(jp.copy()); e_rows.append(t["target"] - p); w_rows.append(np.full(3, w))
                elif ty in ("orient", "root_orient"):
                    b = t.get("body", self.root_body)
                    Rc = d.xmat[b].reshape(3, 3)
                    mujoco.mj_jac(m, d, jp, jr, d.xpos[b], b)
                    e = rotvec_from_mat(t["target"] @ Rc.T)
                    if "axis_w" in t:                      # per-axis weights about the columns of t["axes"]
                        A = t["axes"]
                        J_rows.append(A.T @ jr); e_rows.append(A.T @ e)
                        w_rows.append(w * np.asarray(t["axis_w"], float))
                    else:
                        J_rows.append(jr.copy()); e_rows.append(e); w_rows.append(np.full(3, w))
                elif ty == "com":
                    mujoco.mj_jacSubtreeCom(m, d, jp, self.root_body)
                    k = len(t["target"])
                    J_rows.append(jp[:k].copy()); e_rows.append(t["target"] - d.subtree_com[self.root_body][:k])
                    w_rows.append(np.full(k, w))
                elif ty == "root_height":
                    Jh = np.zeros((1, nv)); Jh[0, 2] = 1.0
                    J_rows.append(Jh); e_rows.append(np.array([t["target"] - q[2]])); w_rows.append(np.array([w]))
                elif ty == "root_xy":
                    Jx = np.zeros((2, nv)); Jx[0, 0] = Jx[1, 1] = 1.0
                    J_rows.append(Jx); e_rows.append(np.asarray(t["target"]) - q[:2]); w_rows.append(np.full(2, w))
                elif ty == "root_yaw":                     # world yaw rate = (R_root @ w_body)_z
                    Jy = np.zeros((1, nv)); Jy[0, 3:6] = d.xmat[self.root_body].reshape(3, 3)[2]
                    dyaw = (t["target"] - yaw_of_quat(q[3:7]) + np.pi) % (2 * np.pi) - np.pi
                    J_rows.append(Jy); e_rows.append(np.array([dyaw])); w_rows.append(np.array([w]))
            J = np.concatenate(J_rows)[:, ia]
            e = np.concatenate(e_rows)
            W = np.concatenate(w_rows)
            err = float(np.sqrt(np.sum(W * e * e)))
            mujoco.mj_differentiatePos(m, dq_post, 1.0, q, post)        # dq that would reach posture
            JW = J.T * W
            H = JW @ J + np.diag(pw[ia] + damping)
            g = JW @ e + pw[ia] * dq_post[ia]
            dqa = np.linalg.solve(H, g)
            n = np.max(np.abs(dqa))
            if n > max_step:
                dqa *= max_step / n
            dq = np.zeros(nv)
            dq[ia] = dqa
            mujoco.mj_integratePos(m, q, dq, 1.0)
            q[self.q_adr] = np.clip(q[self.q_adr], self.q_lo, self.q_hi)
            if n < tol:
                break
        return q, err

    def static_torques(self, qpos, contacts=(True, True), foot_torque_weight=20.0):
        """Joint torques (BODY_JOINTS order) that hold pose `qpos` still against gravity, with the
        body weight carried by the feet flagged in `contacts` (left, right).

        Gravity forces come from inverse dynamics (mj_rne at zero velocity); the part acting on the
        floating base is moved onto the stance feet (minimum-norm split, foot torques penalised so
        weight is shared by forces), then mapped back to the leg joints through the foot Jacobians.
        Add tau / KP to a PD position target to cancel the sag a pure PD loop would have."""
        m, d = self.m, self.d
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        g = np.zeros(self.nv)
        mujoco.mj_rne(m, d, 0, g)                         # generalised gravity force
        feet = [b for b, c in zip(self.foot_ids, contacts) if c]
        if feet:
            jp, jr = self._jacp, self._jacr
            Jc = []
            for b in feet:
                mujoco.mj_jac(m, d, jp, jr, d.xpos[b], b)
                Jc.append(np.vstack([jp, jr]))
            Jc = np.vstack(Jc)                              # (6k, nv): foot [force; torque] -> qfrc
            A = Jc[:, :6].T                                 # base rows
            winv = np.tile([1.0, 1.0, 1.0] + [1.0 / foot_torque_weight] * 3, len(feet))
            F = winv * (A.T @ np.linalg.solve((A * winv) @ A.T + 1e-9 * np.eye(6), g[:6]))
            g = g - Jc.T @ F
        return g[self.v_adr]

    def com(self, qpos=None):
        if qpos is not None:
            self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.subtree_com[self.root_body].copy()

    @property
    def rest_contacts(self):
        """Robot-robot contacts already present in the standing pose (some models have overlapping
        collision shapes at rest); self_collisions() ignores them by default."""
        if not hasattr(self, "_rest_contacts"):
            self._rest_contacts = set()
            self._rest_contacts = self.self_collisions(self.standing_qpos(), ignore=())
        return self._rest_contacts

    def self_collisions(self, qpos, ignore=None):
        """Robot-robot contact pairs (geom ids) at qpos, excluding pairs listed in `ignore`
        (default: the ones already touching in the standing pose)."""
        if ignore is None:
            ignore = self.rest_contacts
        m, d = self.m, self.d
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        geoms = _contact_geoms(d)
        body = m.geom_bodyid
        out = set()
        for g1, g2 in geoms:
            if body[g1] != 0 and body[g2] != 0:
                pair = (min(g1, g2), max(g1, g2))
                if pair not in ignore:
                    out.add(pair)
        return out


def _contact_geoms(d):
    n = d.ncon
    if n == 0:
        return np.zeros((0, 2), int)
    c = d.contact
    if hasattr(c, "geom"):
        g = np.asarray(c.geom)
        if g.ndim == 2:
            return g[:n]
    return np.stack([np.asarray(c.geom1)[:n], np.asarray(c.geom2)[:n]], 1)


class G1Kinematics(Kinematics):
    """Kinematics of the G1 (kept for code written before other robots were added)."""

    def __init__(self, xml_path=None):
        super().__init__("g1", xml_path)
