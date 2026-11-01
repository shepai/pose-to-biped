"""
G1 keypoints, forward kinematics and a whole-body IK solver (no torch, plain MuJoCo + numpy).

Keypoint frame ("heading frame")
--------------------------------
All keypoint targets the env and the retargeting tools exchange are expressed in the robot's
heading frame:  origin = pelvis (x, y) projected onto the ground (z = 0 is the floor),
axes = pelvis yaw only (x forward, y left, z up).  So a keypoint's z is its height above the
floor, and x/y are relative to the pelvis.  This is also the natural frame for a body tracker:
take each joint relative to the hips' mid-point, rotate by the body's heading, add the pelvis
height, scale by robot_size / human_size.
"""
from pathlib import Path

import numpy as np
import mujoco

from G1_tele.envs.whole_body import BODY_JOINTS, DEFAULT_Q, XML_PATH

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
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
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


# ---------------------------------------------------------------- kinematic model
class G1Kinematics:
    """A private MjModel/MjData used purely for kinematics (FK, Jacobians, IK) - never stepped."""

    def __init__(self, xml_path=XML_PATH):
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.nq, self.nv = m.nq, m.nv
        self.q_adr = np.array([m.joint(n).qposadr[0] for n in BODY_JOINTS])
        self.v_adr = np.array([m.joint(n).dofadr[0] for n in BODY_JOINTS])
        self.q_lo = np.array([m.joint(n).range[0] for n in BODY_JOINTS])
        self.q_hi = np.array([m.joint(n).range[1] for n in BODY_JOINTS])
        self.pelvis_id = m.body("pelvis").id
        self.root_body = self.pelvis_id
        self.foot_ids = [m.body(n).id for n in FEET]
        self._kp_cache = {}
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))

    # ---------- state <-> arrays
    def qpos_from(self, root, joints):
        q = self.m.qpos0.copy()
        q[:7] = root
        q[self.q_adr] = joints
        return q

    def default_qpos(self, height=0.793):
        return self.qpos_from(np.array([0, 0, height, 1, 0, 0, 0.0]), DEFAULT_Q)

    def keypoint_spec(self, names):
        key = tuple(names)
        if key not in self._kp_cache:
            self._kp_cache[key] = (np.array([self.m.body(KEYPOINTS[n][0]).id for n in names]),
                                   np.array([KEYPOINTS[n][1] for n in names], float))
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

    def self_collisions(self, qpos, ignore=()):
        """Robot-robot contact pairs (geom ids) at qpos, excluding pairs listed in `ignore`."""
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
