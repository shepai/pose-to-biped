"""
src/G1_tele/envs/pose_tracking.py

Whole-body pose tracking for a humanoid (the 29-DoF G1 by default; robot="h1", "t1", "n1" or any
robot in G1_tele.robots): follow keypoint targets (hands, feet, and optionally elbows, knees, hips,
shoulders) while keeping balance. One policy controls every joint - there is no frozen walking
policy underneath, because a stepping gait cannot hold a foot where you ask it.

Robots and layouts
------------------
  * robot: which humanoid (default "g1"). Its motion library defaults to RobotSpec.motion_file
    (make it with examples/RL/make_motions.py synth --robot <name>). Lengths in the reward and the
    termination tests (sigma_kp, max_kp_err, max_root_err, the capture point and height tolerances)
    are given for the G1 and scaled by the robot's leg length; pushes by its mass.
  * layout="native" (default): observations and actions list the robot's own joints. The G1's are
    exactly what they always were, so older G1 models keep working.
  * layout="universal": every per-joint entry is laid out over the 31 joint slots shared by all
    robots (G1_tele.robots.spec.SLOTS; zero where a robot has no such joint or doesn't drive it),
    angles signed so that each slot turns the same way on every robot, lengths divided by the
    robot's scale, and a robot descriptor (which slots it drives, its scale, hip width and mass) is
    added. The action has 31 entries; the ones for slots the robot doesn't drive are ignored. Every
    robot then has the same observation and action size, so ONE policy can be trained on several
    robots at once (examples/RL/train_pose_tracking.py --robot g1,h1,t1,n1).

Balance comes first, pose accuracy second
-----------------------------------------
  * Reward = alive bonus + balance terms + tracking terms - small penalties (REWARD_WEIGHTS;
    override any of them with reward_weights={...}). With the defaults, staying up and balanced
    is worth 60 % of the per-step reward, matching the pose 40 %.
      balance:  capture point inside the support polygon of the feet on the ground (the term that
                warns about a fall ~0.5 s before it happens), pelvis tilt, pelvis height, low
                roll/pitch rate, feet on the ground when the reference has them planted
      tracking: keypoints (sigma_kp tolerance), joint angles, joint velocities
  * Privileged critic: the observation ends with balance state the real robot cannot measure
    (base velocity, centre of mass, capture point margin, foot contacts, friction, pushes). The
    PoseTrackingPolicy actor only sees the first n_actor_obs entries; the critic sees all of it,
    so it can predict falls and give the actor much cleaner learning signals.
  * Curriculum hooks: set_kind_weights() changes which clip kinds are sampled, and every episode
    reports info["clip_kind"] and info["survival"] (fraction of the possible episode survived).
    examples/run_logging.CurriculumCallback uses them to unlock harder motions as balance improves.

Other parts of the recipe (DeepMimic / H2O, scaled down for SB3):
  * Reference motions come from a motion library (G1_tele.motion.library): full robot
    configurations, by default procedurally generated balanced poses (examples/make_motions.py),
    or keypoint clips you retarget with G1_tele.motion.retarget.
  * Reference state initialisation: each episode starts at a random time of a random clip with the
    robot placed exactly in the reference pose.
  * Early termination: when the robot falls or the mean keypoint error exceeds max_kp_err.
  * Gravity feedforward (gravity_ff=True, residual mode): the PD target also gets tau_static / KP,
    the torques that hold the reference pose against gravity with the weight on the stance feet.
  * Balance assist (optional, off by default): stabilising wrench on the pelvis, annealed to zero.

Action (native: n_ctrl values in [-1, 1], n_ctrl = the robot's driven joints, for the G1 23 with
wrists locked, 29 otherwise; universal: 31 values, one per slot):
    residual=True   PD target = reference joints (+ gravity offset) + action_scale * action  (default)
    residual=False  PD target = default pose + action_scale * action
Observation = [actor part (n_actor_obs) | privileged part (N_PRIVILEGED, only if privileged_obs)]
  actor part (all positions in the heading frame, see G1_tele.motion.kinematics):
    ang_vel*0.25, gravity, (q - default), dq*0.05, last action            (robot, proprioceptive)
    reference keypoints, reference - robot keypoints                      (targets for the next step)
    robot pelvis height, reference pelvis height, reference gravity vector
    [residual] reference joints - default, reference joints - q, reference joint velocity*0.05
    [lookahead] reference keypoints (and joints) at t + each lookahead time
    [locomotion] reference root velocity (forward, sideways, turning rate) now and at each lookahead
                 time, and the robot's position and heading error to the reference root
Deploying: feed the actor obs[:n_actor_obs] (the privileged tail can be zeros - the actor ignores
it). residual=True needs reference joint angles at run time: run
G1_tele.motion.retarget.KeypointRetargeter on the tracker keypoints (~2 ms per frame).

Locomotion (locomotion=True)
----------------------------
Without it, everything is measured in each body's own heading frame, so the robot is never asked to
go anywhere: a reference that walks or turns scores the same whether the robot follows or stands
still. With locomotion=True the reference root's motion becomes part of the task:
  * observations: the reference's root velocity in its heading frame and its turning rate (now and at
    each lookahead time), and where the reference root is relative to the robot (position and heading
    error, clipped to the termination limits),
  * rewards (LOCO_REWARD_WEIGHTS): following the root velocity and turning rate, plus weaker terms
    for the position and heading error so drift is corrected,
  * the foot-slip penalty is measured in the world, not relative to the pelvis (walking moves the
    pelvis over a planted foot, which the heading-frame version would punish),
  * termination: the episode also ends ("lost") when the robot is more than max_root_err metres or
    max_yaw_err radians away from the reference root. This is what makes ignoring the motion costly:
    a robot that stands still on a walking clip is stopped within a couple of seconds.
The observation grows by 6 + 3 * len(lookahead) entries, inserted after the old actor part, so a
policy trained without locomotion can be widened to it (examples/RL/train_locomotion.py).
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from G1_tele.envs.whole_body import HumanoidEnv
from G1_tele.motion.kinematics import (CONTACT_GAP, KEYPOINT_SETS, Kinematics, _g1_geometry, quat_to_mat, rot_z,
                                       rotvec_from_mat, yaw_of_quat)
from G1_tele.motion.library import MotionLibrary, default_motions
from G1_tele.robots import get_robot
from G1_tele.robots.spec import N_SLOTS, ABSOLUTE_SCALE as _ABS_SLOTS, RESIDUAL_SCALE as _RES_SLOTS

# residual / absolute action range (rad) per joint slot - the G1's 29 joints are slots 0-28
RESIDUAL_SCALE, ABSOLUTE_SCALE = _RES_SLOTS[:29].copy(), _ABS_SLOTS[:29].copy()
LAYOUTS = ("native", "universal")
N_ROBOT_DESC = N_SLOTS + 3    # universal layout: driven-slot mask, scale, lateral scale, mass / G1 mass

KEYPOINT_WEIGHT = {"hand": 1.0, "foot": 1.0, "toe": 0.5, "elbow": 0.5, "knee": 0.5,
                   "shoulder": 0.5, "hip": 0.5, "head": 0.5}

# Positive terms sum to 1.0: alive 0.20 + balance 0.40 + tracking 0.40.
REWARD_WEIGHTS = dict(
    alive=0.20,
    cp=0.15,          # capture point inside the support polygon (with a 2 cm margin)
    upright=0.10,     # pelvis tilt matches the reference
    height=0.05,      # pelvis height matches the reference
    ang_vel=0.05,     # little pelvis roll/pitch rate (no wobbling)
    contact=0.05,     # feet on the ground when the reference has them planted (and lifted when lifted)
    kp=0.20,          # keypoint positions
    joint=0.12,       # joint angles
    joint_vel=0.08,   # joint velocities
    action_rate=0.08, torque=1e-6, slip=0.05,                 # penalties
)
# With locomotion=True: alive 0.20 + balance 0.34 + pose 0.26 + root motion 0.20.
LOCO_REWARD_WEIGHTS = dict(
    alive=0.20,
    cp=0.12, upright=0.08, height=0.04, ang_vel=0.04, contact=0.06,
    kp=0.15, joint=0.07, joint_vel=0.04,
    lin_vel=0.09,     # pelvis velocity (heading frame) matches the reference root's
    yaw_rate=0.05,    # turning rate matches
    root_pos=0.04,    # pelvis position on the floor matches the reference root (drift correction)
    yaw=0.02,         # heading matches
    action_rate=0.08, torque=1e-6, slip=0.05,
)
SIGMAS = dict(cp=0.04, cp_margin=0.02, tilt=0.15, height=0.05, ang_vel=1.0, joint=0.3, joint_vel=1.5,
              lin_vel=0.2, yaw_rate=0.4, root_pos=0.25, yaw=0.3)
_LENGTH_SIGMAS = ("cp", "cp_margin", "height", "lin_vel", "root_pos")     # scaled with the robot's size

N_PRIVILEGED = 19
CONTACT_Z = 0.010            # G1: a sole corner sphere (r = 5 mm) closer than this to the floor is in contact
LEFT_RGBA, RIGHT_RGBA, MID_RGBA = [1.0, 0.25, 0.2, 0.75], [0.2, 0.45, 1.0, 0.75], [0.2, 0.9, 0.3, 0.75]


def _names(spec):
    return list(KEYPOINT_SETS[spec]) if isinstance(spec, str) else list(spec)


# ---------------------------------------------------------------- 2-D support polygon helpers
def convex_hull(pts):
    """Monotone chain; pts (n,2) -> hull vertices counter-clockwise (n <= 8 here)."""
    p = sorted(map(tuple, pts))
    if len(p) <= 2:
        return np.array(p)

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    lower, upper = half(p), half(reversed(p))
    return np.array(lower[:-1] + upper[:-1])


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def support_margin(p, hull):
    """Signed distance from point p to the polygon: > 0 inside (how far from the edge), < 0 outside."""
    n = len(hull)
    if n == 0:
        return -1.0
    if n == 1:
        return -float(np.linalg.norm(p - hull[0]))
    if n == 2:
        return -_seg_dist(p, hull[0], hull[1])
    inside, dmin = True, np.inf
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0:
            inside = False
        dmin = min(dmin, _seg_dist(p, a, b))
    return dmin if inside else -dmin


class PoseTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, motion_file=None, xml_path=None, keypoints="full", obs_keypoints=None,
                 residual=True, control_wrists=False, episode_seconds=10.0, lookahead=(),
                 kind_weights=None, max_kp_err=0.30, push_force=0.0, push_interval=(2.0, 6.0),
                 friction_range=None, init_noise=0.03, obs_noise=0.0, action_scale=None,
                 sigma_kp=0.12, reward_weights=None, privileged_obs=True, gravity_ff=True,
                 assist=0.0, assist_anneal_steps=None, locomotion=False, max_root_err=0.5, max_yaw_err=0.8,
                 render_mode=None, seed=None, robot=None, layout="native"):
        super().__init__()
        spec = get_robot(robot)
        if xml_path is not None and str(xml_path) != str(spec.xml_path):
            spec = spec.with_xml(xml_path)
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        self.spec, self.robot, self.layout = spec, spec.name, layout
        self.universal = layout == "universal"
        self.base = HumanoidEnv(robot=spec, render_mode=render_mode, max_episode_seconds=1e9)
        self.kin = Kinematics(spec)                         # reference poses live here (never stepped)
        if hasattr(motion_file, "frame") and hasattr(motion_file, "kinds"):
            self.lib = motion_file                          # a ready-made reference (e.g. envs.live.LiveReference)
        else:
            self.lib = MotionLibrary(motion_file or default_motions(spec), robot=spec.name)
        self.clip_kinds = sorted(set(str(k) for k in self.lib.kinds))
        k = self.kin
        sc = k.scale                                        # G1 lengths -> this robot's
        self.mass_ratio = k.mass / _g1_geometry()["mass"]   # push forces and the assist are given for the G1
        self.L = sc if self.universal else 1.0              # length unit of the observation

        self.kp_names = _names(keypoints)
        self.obs_kp_names = _names(obs_keypoints) if obs_keypoints is not None else self.kp_names
        self.kp_w = np.array([next(w for k_, w in KEYPOINT_WEIGHT.items() if k_ in n) for n in self.kp_names])
        all_names = self.kp_names + [n for n in self.obs_kp_names if n not in self.kp_names]
        missing = [n for n in all_names if n not in k.keypoints]
        if missing:
            raise ValueError(f"the {spec.name} has no keypoints {missing}")
        self._all_kp = all_names
        self._obs_sel = [all_names.index(n) for n in self.obs_kp_names]
        self.foot_kp = [all_names.index(n) for n in ("left_foot", "right_foot") if n in all_names]

        self.residual = residual
        self.gravity_ff = gravity_ff and residual
        nj = spec.n_joints
        self.n_joints = nj
        self.ctrl = np.arange(nj) if control_wrists else np.array(spec.ctrl_idx)
        self.n_ctrl = len(self.ctrl)
        scale = spec.residual_scale if residual else spec.absolute_scale
        self.action_scale = (np.asarray(action_scale, float) * np.ones(nj) if action_scale is not None
                             else scale.copy())[self.ctrl]
        # universal layout: joint i lives in slot slot_idx[i], with sign signs[i]
        self.slot_idx = spec.slot_idx
        self.signs = k.joint_signs()
        self.slot_mask = np.zeros(N_SLOTS, np.float32)
        self.slot_mask[self.slot_idx[self.ctrl]] = 1.0
        self.robot_desc = np.concatenate([self.slot_mask, [sc, k.lateral_scale, self.mass_ratio]]).astype(np.float32)
        self.episode_seconds, self.lookahead = episode_seconds, tuple(lookahead)
        self.kind_weights = dict(kind_weights) if kind_weights else None
        self.max_kp_err, self.sigma_kp = max_kp_err * sc, sigma_kp * sc
        self.locomotion = bool(locomotion)
        self.max_root_err, self.max_yaw_err = float(max_root_err) * sc, float(max_yaw_err)
        self.sg = {key: v * (sc if key in _LENGTH_SIGMAS else 1.0) for key, v in SIGMAS.items()}
        self.fall_drop = 0.25 * sc
        self.end_off_course = True                          # LivePoseTrackingEnv turns this off (it re-anchors instead)
        base_w = LOCO_REWARD_WEIGHTS if self.locomotion else REWARD_WEIGHTS
        unknown = set(reward_weights or {}) - set(base_w)
        if unknown:
            raise ValueError(f"unknown reward_weights keys {unknown}; valid: {list(base_w)}")
        self.w = {**base_w, **(reward_weights or {})}
        self.push_force, self.push_interval = push_force * self.mass_ratio, push_interval
        self.friction_range, self.init_noise, self.obs_noise = friction_range, init_noise, obs_noise
        self.privileged_obs = privileged_obs
        self.assist0, self.assist_anneal_steps = float(assist), assist_anneal_steps
        self.assist = self.assist0
        self.total_steps = 0                                # env steps taken over this env's lifetime
        self.push_wrench = np.zeros(6)
        self.friction_scale = 1.0

        m = self.base.m
        self.pelvis_id = self.base.root_id                  # the floating base (the pelvis on most robots)
        self._kp_ids = np.array([m.body(k.keypoints[n][0]).id for n in all_names])
        self._kp_off = np.array([k.keypoints[n][1] for n in all_names], float)
        self._friction0 = m.geom_friction[:, 0].copy()
        self.q_lo, self.q_hi = self.base.q_lo + 0.02, self.base.q_hi - 0.02
        self._foot_ids = [m.body(nm).id for nm in spec.feet]
        self._sole = [np.asarray(sp, float) for sp in k.sole_points]     # (n, 4): x, y, z, r in the foot frame

        K = len(self.obs_kp_names)
        n = N_SLOTS if self.universal else self.n_ctrl
        self.n_pose_obs = (6 + 3 * n + 6 * K + 5 + (3 * n if residual else 0)
                           + len(self.lookahead) * (3 * K + (n if residual else 0))
                           + (N_ROBOT_DESC if self.universal else 0))
        self.n_loco_obs = (6 + 3 * len(self.lookahead)) if self.locomotion else 0
        self.n_actor_obs = self.n_pose_obs + self.n_loco_obs
        n_obs = self.n_actor_obs + (N_PRIVILEGED if privileged_obs else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.render_mode = render_mode
        self.reset(seed=seed)

    # ------------------------------------------------------------ curriculum hooks
    def set_kind_weights(self, weights):
        """Sampling weight per clip kind for the next episodes, e.g. {"upper": 1, "whole": 1}."""
        self.kind_weights = dict(weights) if weights else None

    def set_assist(self, value):
        """Fix the balance-assist strength (0 = off) and stop the automatic annealing."""
        self.assist0, self.assist, self.assist_anneal_steps = float(value), float(value), None

    @property
    def kind_seconds(self):
        """Seconds of usable motion (clips >= 1.5 s, the shortest the env samples) per clip kind."""
        out = {}
        for c in range(self.lib.n_clips):
            d = self.lib.duration(c)
            if d >= 1.5:
                k = str(self.lib.kinds[c])
                out[k] = out.get(k, 0.0) + d
        return out

    # ------------------------------------------------------------ reference
    def _ref(self, t):
        """Reference state at time t: root, joints, jvel, local keypoints, height, gravity, foot contacts."""
        lib, c = self.lib, self.clip
        root, joints = lib.frame(c, t)
        _, _, jvel = lib.velocity(c, t)
        k = self.kin
        k.forward(k.qpos_from(root, joints))
        o, R = k.heading(root[:3], root[3:7])
        kps = (k.keypoints_world(self._all_kp) - o) @ R
        contact = k.d.xpos[k.foot_ids, 2] < k.planted_height
        Rq = quat_to_mat(root[3:7])
        g = Rq.T @ np.array([0.0, 0.0, -1.0])
        ref = {"root": root, "joints": joints, "jvel": jvel, "kp": kps, "h": root[2], "g": g, "contact": contact}
        if self.locomotion:
            lin, ang, _ = lib.velocity(c, t)
            ref["v"] = (lin @ R)[:2]                        # root velocity in the reference's heading frame
            ref["wz"] = float((Rq @ ang)[2])                # turning rate (world z)
            ref["xy"] = np.asarray(root[:2], float).copy()
            ref["yaw"] = float(yaw_of_quat(root[3:7]))
        return ref

    def _root_error(self, ref):
        """Reference root relative to the robot: (dx, dy) in the robot's heading frame, heading error."""
        d = self.base.d
        yaw = yaw_of_quat(d.qpos[3:7])
        e = (ref["xy"] - d.qpos[:2]) @ rot_z(yaw)[:2, :2]
        dyaw = (ref["yaw"] - yaw + np.pi) % (2 * np.pi) - np.pi
        return e, float(dyaw)

    def _gravity_offset(self, ref):
        """PD-target offset (tau_static / KP) that cancels gravity sag in the reference pose."""
        k = self.kin
        contacts = tuple(ref["contact"]) if ref["contact"].any() else (True, True)
        tau = k.static_torques(k.qpos_from(ref["root"], ref["joints"]), contacts)
        tau = np.clip(tau, -self.base.tau_lim, self.base.tau_lim)
        return np.clip(tau / self.base.kp, -0.5, 0.5)

    # ------------------------------------------------------------ robot
    def _robot_kp(self):
        d = self.base.d
        R = d.xmat[self._kp_ids].reshape(-1, 3, 3)
        p = d.xpos[self._kp_ids] + np.einsum("kij,kj->ki", R, self._kp_off)
        o = np.array([d.qpos[0], d.qpos[1], 0.0])
        return (p - o) @ rot_z(yaw_of_quat(d.qpos[3:7]))

    def _heading_to_world(self, p_local):
        d = self.base.d
        return np.array([d.qpos[0], d.qpos[1], 0.0]) + p_local @ rot_z(yaw_of_quat(d.qpos[3:7])).T

    def _balance_state(self):
        """CoM, CoM velocity, capture point margin and foot contacts. Needs kinematics + comPos done."""
        d = self.base.d
        com = d.subtree_com[self.pelvis_id].copy()
        com_vel = (com - self.prev_com) / self.base.dt
        self.prev_com = com
        pts, contact = [], []
        for b, sole in zip(self._foot_ids, self._sole):
            p = d.xpos[b] + sole[:, :3] @ d.xmat[b].reshape(3, 3).T
            on = p[:, 2] - sole[:, 3] < CONTACT_GAP
            contact.append(bool(on.any()))
            pts.append(p[on, :2])
        pts = np.concatenate(pts)
        omega0 = np.sqrt(9.81 / max(com[2], 0.3 * self.kin.scale))
        cp = com[:2] + com_vel[:2] / omega0
        margin = support_margin(cp, convex_hull(pts)) if len(pts) else -0.3
        self.bal = {"com": com, "com_vel": com_vel, "cp": cp, "margin": margin, "contact": np.array(contact)}
        return self.bal

    # ------------------------------------------------------------ obs
    def to_slots(self, x):
        """Per-joint values (n_joints,) -> the universal layout (31,): signed, in the joints' slots,
        zero for slots this robot doesn't drive."""
        out = np.zeros(N_SLOTS)
        c = self.ctrl
        out[self.slot_idx[c]] = self.signs[c] * np.asarray(x, float)[c]
        return out

    def from_slots(self, a):
        """A universal action (31,) -> this robot's action (n_ctrl,)."""
        c = self.ctrl
        return self.signs[c] * np.asarray(a, float)[self.slot_idx[c]]

    def _obs(self):
        b, s = self.base, self.base.state()
        c, rn = self.ctrl, self.ref_next
        J = self.to_slots if self.universal else (lambda x: x[c])     # per-joint entries
        L = self.L
        kp_now = self.robot_kp[self._obs_sel]
        kp_ref = rn["kp"][self._obs_sel]
        last = self.last_action * self.slot_mask if self.universal else self.last_action
        parts = [s["ang_vel"] * 0.25, s["gravity"], J(s["q"] - b.default_q), J(s["dq"]) * 0.05, last,
                 kp_ref.ravel() / L, (kp_ref - kp_now).ravel() / L,
                 [b.d.qpos[2] / L, rn["h"] / L], rn["g"]]
        if self.residual:
            parts += [J(rn["joints"] - b.default_q), J(rn["joints"] - s["q"]), J(rn["jvel"]) * 0.05]
        ahead = []
        for dt in self.lookahead:
            r = self._ref(self.t + b.dt + dt)
            ahead.append(r)
            parts.append(r["kp"][self._obs_sel].ravel() / L)
            if self.residual:
                parts.append(J(r["joints"] - b.default_q))
        if self.universal:                                  # which robot this is (end of the pose part)
            parts.append(self.robot_desc)
        if self.locomotion:                                 # after the pose part, so old policies can be widened
            parts.append([rn["v"][0] / L, rn["v"][1] / L, rn["wz"]])
            for r in ahead:
                parts.append([r["v"][0] / L, r["v"][1] / L, r["wz"]])
            e, dyaw = self._root_error(rn)
            parts.append(np.clip(e, -self.max_root_err, self.max_root_err) / L)
            parts.append([np.clip(dyaw, -self.max_yaw_err, self.max_yaw_err)])
        obs = np.concatenate(parts).astype(np.float32)
        if self.obs_noise > 0:
            obs += self.np_random.normal(0, self.obs_noise, obs.shape).astype(np.float32)
        if self.privileged_obs:
            d, bal = b.d, self.bal
            Rh = rot_z(yaw_of_quat(d.qpos[3:7]))
            o = np.array([d.qpos[0], d.qpos[1], 0.0])
            M = self.mass_ratio if self.universal else 1.0
            priv = np.concatenate([d.qvel[:3] @ Rh / L, (bal["com"] - o) @ Rh / L, bal["com_vel"] @ Rh / L,
                                   [np.clip(bal["margin"] / L, -0.3, 0.3) * 10], bal["contact"], rn["contact"],
                                   [self.friction_scale], self.push_wrench[:3] / (100.0 * M), [self.assist]])
            obs = np.concatenate([obs, priv.astype(np.float32)])
        return obs

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        b, rng = self.base, self.np_random
        b.reset()
        if options and "clip" in options:
            self.clip = int(options["clip"])
        else:
            self.clip = self.lib.sample_clip(rng, self.kind_weights, min_seconds=1.5)
        dur = self.lib.duration(self.clip)
        t0 = float(options["t0"]) if options and "t0" in options else float(rng.uniform(0, max(0.0, dur - 1.0)))
        self.t, self.t_end = t0, min(t0 + self.episode_seconds, dur)
        self.max_ep_steps = max(1, int(round((self.t_end - t0) / b.dt)))

        # reference state initialisation
        ref = self._ref(t0)
        lin, ang, jvel = self.lib.velocity(self.clip, t0)
        d = b.d
        d.qpos[:7] = ref["root"]
        d.qpos[2] += 0.002
        d.qpos[b.q_adr] = np.clip(ref["joints"] + rng.normal(0, self.init_noise, self.n_joints), b.q_lo, b.q_hi)
        d.qvel[:3], d.qvel[3:6] = lin, ang
        d.qvel[b.v_adr] = jvel
        mujoco.mj_forward(b.m, d)

        if self.friction_range is not None:
            self.friction_scale = float(rng.uniform(*self.friction_range))
            b.m.geom_friction[:, 0] = self._friction0 * self.friction_scale
        self.next_push = rng.uniform(*self.push_interval) if self.push_force > 0 else np.inf
        self.push_left = 0
        self.push_wrench[:] = 0.0
        b.d.xfrc_applied[:] = 0.0
        self.last_action = np.zeros(self.action_space.shape[0], np.float32)
        self._last_native = np.zeros(self.n_ctrl, np.float32)
        self.robot_kp = self._robot_kp()
        self.prev_foot = self.robot_kp[self.foot_kp].copy()
        self.prev_foot_world = self._heading_to_world(self.prev_foot)
        self.prev_com = d.subtree_com[self.pelvis_id] - np.asarray(d.qvel[:3]) * b.dt  # com_vel ~ base vel
        self._balance_state()
        self.ref_next = self._ref(self.t + b.dt)
        self.ep_stats = {"kp_err": 0.0, "steps": 0}
        return self._obs(), {}

    def _push(self):
        if self.push_left > 0:
            self.push_left -= 1
            if self.push_left == 0:
                self.push_wrench[:3] = 0.0
        elif self.t >= self.next_push:
            ang = self.np_random.uniform(0, 2 * np.pi)
            f = self.push_force * self.np_random.uniform(0.3, 1.0)
            self.push_wrench[:3] = f * np.array([np.cos(ang), np.sin(ang), 0.0])
            self.push_left = int(self.np_random.integers(5, 11))
            self.next_push = self.t + self.np_random.uniform(*self.push_interval)

    def _assist_wrench(self, ref):
        """'Hand of god' on the pelvis: PD towards the reference pelvis height and roll/pitch.
        Yaw and horizontal position are never assisted."""
        d, w = self.base.d, np.zeros(6)
        if self.assist <= 0:
            return w
        R = quat_to_mat(d.qpos[3:7])
        yaw = yaw_of_quat(d.qpos[3:7])
        R_t = rot_z(yaw) @ rot_z(yaw_of_quat(ref["root"][3:7])).T @ quat_to_mat(ref["root"][3:7])
        e = rotvec_from_mat(R_t @ R.T)
        e[2] = 0.0
        omega = R @ d.qvel[3:6]
        w[3:5] = 250.0 * e[:2] - 15.0 * omega[:2]
        w[2] = 1500.0 * (ref["h"] - d.qpos[2]) - 100.0 * d.qvel[2]
        return self.assist * self.mass_ratio * w

    def step(self, action):
        b = self.base
        self.total_steps += 1
        if self.assist_anneal_steps:
            self.assist = self.assist0 * max(0.0, 1.0 - self.total_steps / self.assist_anneal_steps)
        a_in = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
        a = self.from_slots(a_in).astype(np.float32) if self.universal else a_in
        ref = self.ref_next                                 # reference at the end of this control step
        target = (ref["joints"] if self.residual else b.default_q).astype(np.float64).copy()
        if self.gravity_ff:
            target += self._gravity_offset(ref)
        target[self.ctrl] += self.action_scale * a
        target = np.clip(target, self.q_lo, self.q_hi)

        self._push()
        if self.assist > 0:
            pid = self.pelvis_id

            def apply(d):                                   # 500 Hz so the stiff assist PD stays stable
                d.xfrc_applied[pid] = self.push_wrench + self._assist_wrench(ref)
            b.step_targets(target, apply)
        else:
            b.d.xfrc_applied[self.pelvis_id] = self.push_wrench
            b.step_targets(target)
        mujoco.mj_kinematics(b.m, b.d)
        mujoco.mj_comPos(b.m, b.d)
        self.t += b.dt
        self.robot_kp = kp = self._robot_kp()
        s = b.state()
        bal = self._balance_state()
        w, sg = self.w, self.sg

        # ---- balance terms (robot now vs reference at the same instant)
        r_cp = np.exp(-(max(0.0, sg["cp_margin"] - bal["margin"]) / sg["cp"]) ** 2)
        dg = np.linalg.norm(s["gravity"] - ref["g"])
        r_up = np.exp(-(dg / sg["tilt"]) ** 2)
        dh = b.d.qpos[2] - ref["h"]
        r_h = np.exp(-(dh / sg["height"]) ** 2)
        r_av = np.exp(-np.sum(s["ang_vel"][:2] ** 2) / sg["ang_vel"] ** 2)
        contact_match = float(np.mean(bal["contact"] == ref["contact"]))

        # ---- tracking terms
        c = self.ctrl
        nk = len(self.kp_names)
        e_kp = np.linalg.norm(kp[:nk] - ref["kp"][:nk], axis=1)
        kp_err = float(np.sum(self.kp_w * e_kp) / np.sum(self.kp_w))
        r_kp = np.exp(-np.sum(self.kp_w * e_kp ** 2) / np.sum(self.kp_w) / self.sigma_kp ** 2)
        e_q = s["q"][c] - ref["joints"][c]
        r_q = np.exp(-np.mean(e_q ** 2) / sg["joint"] ** 2)
        r_vel = np.exp(-np.mean((s["dq"][c] - ref["jvel"][c]) ** 2) / sg["joint_vel"] ** 2)

        # ---- penalties: planted reference feet should not slide, smooth actions, low torque
        foot = kp[self.foot_kp]
        foot_world = self._heading_to_world(foot)
        slip = 0.0
        for i, fi in enumerate(self.foot_kp):
            if ref["kp"][fi, 2] < self.kin.planted_height:
                if self.locomotion:                         # in the world: the pelvis may move over the foot
                    if bal["contact"][i]:
                        slip += np.sum(((foot_world[i, :2] - self.prev_foot_world[i, :2]) / b.dt) ** 2)
                else:
                    slip += np.sum(((foot[i, :2] - self.prev_foot[i, :2]) / b.dt) ** 2)
        self.prev_foot = foot.copy()
        self.prev_foot_world = foot_world
        p_rate = np.mean((a - self._last_native) ** 2)
        p_tau = np.sum(b.d.ctrl[b.act_adr] ** 2)

        r_balance = w["cp"] * r_cp + w["upright"] * r_up + w["height"] * r_h + w["ang_vel"] * r_av \
            + w["contact"] * contact_match
        r_track = w["kp"] * r_kp + w["joint"] * r_q + w["joint_vel"] * r_vel
        loco_info, off_course = {}, False
        if self.locomotion:                                 # ---- root motion
            d = b.d
            Rh = rot_z(yaw_of_quat(d.qpos[3:7]))
            v = (d.qvel[:3] @ Rh)[:2]
            wz = float((quat_to_mat(d.qpos[3:7]) @ d.qvel[3:6])[2])
            e_v = float(np.linalg.norm(v - ref["v"]))
            e_wz = abs(wz - ref["wz"])
            e_xy, dyaw = self._root_error(ref)
            e_p = float(np.linalg.norm(e_xy))
            r_track += (w["lin_vel"] * np.exp(-(e_v / sg["lin_vel"]) ** 2)
                        + w["yaw_rate"] * np.exp(-(e_wz / sg["yaw_rate"]) ** 2)
                        + w["root_pos"] * np.exp(-(e_p / sg["root_pos"]) ** 2)
                        + w["yaw"] * np.exp(-(dyaw / sg["yaw"]) ** 2))
            off_course = e_p > self.max_root_err or abs(dyaw) > self.max_yaw_err
            loco_info = {"vel_err": e_v, "yaw_rate_err": e_wz, "root_err": e_p, "yaw_err": abs(dyaw),
                         "ref_speed": float(np.linalg.norm(ref["v"])), "off_course": off_course}
        rew = w["alive"] + r_balance + r_track - w["action_rate"] * p_rate - w["torque"] * p_tau - w["slip"] * slip
        self.last_action = a_in
        self._last_native = a

        # fell: base tilted > ~45 deg, or 25 cm (G1; x scale) lower than the reference; lost: pose too
        # far off (or, with locomotion, too far from the reference root's position or heading)
        fell = bool(s["gravity"][2] > -0.7 or dh < -self.fall_drop)
        lost = (not fell) and (kp_err > self.max_kp_err or (off_course and self.end_off_course))
        terminated = fell or lost
        truncated = (not terminated) and self.t >= self.t_end - 1e-6

        self.ep_stats["kp_err"] += kp_err
        self.ep_stats["steps"] += 1
        names = self.kp_names
        info = {"kp_err": kp_err, "joint_err": float(np.sqrt(np.mean(e_q ** 2))), "height_err": float(abs(dh)),
                "hand_err": float(np.mean([e_kp[i] for i, n in enumerate(names) if "hand" in n] or [np.nan])),
                "foot_err": float(np.mean([e_kp[i] for i, n in enumerate(names) if "foot" in n] or [np.nan])),
                "cp_margin": float(bal["margin"]), "contact_match": contact_match,
                "r_balance": float(r_balance), "r_track": float(r_track),
                "r_cp": float(r_cp), "r_upright": float(r_up), "r_kp": float(r_kp), "r_joint": float(r_q),
                "fell": fell, "lost": lost, "assist": self.assist, **loco_info}
        if terminated or truncated:
            info["episode_kp_err"] = self.ep_stats["kp_err"] / max(1, self.ep_stats["steps"])
            info["clip_kind"] = str(self.lib.kinds[self.clip])
            info["robot"] = self.robot
            info["survival"] = 1.0 if not terminated else min(1.0, self.ep_stats["steps"] / self.max_ep_steps)
        self.ref_next = self._ref(self.t + b.dt)
        return self._obs(), float(rew), terminated, truncated, info

    # ------------------------------------------------------------ viewer / recording helpers
    @property
    def model(self):
        return self.base.m

    @property
    def data(self):
        return self.base.d

    @property
    def dt(self):
        return self.base.dt

    @property
    def frame_skip(self):
        return self.base.decim

    def current_targets_world(self):
        """Reference keypoints placed on the robot (the robot's heading frame), world coordinates."""
        r = self._ref(self.t)
        return self._heading_to_world(r["kp"][:len(self.kp_names)])

    def draw_targets(self, scene, radius=0.03):
        """Add target spheres (left = red, right = blue) to an mjvScene (viewer.user_scn or Renderer.scene)."""
        for name, p in zip(self.kp_names, self.current_targets_world()):
            if scene.ngeom >= scene.maxgeom:
                break
            rgba = LEFT_RGBA if name.startswith("left") else RIGHT_RGBA if name.startswith("right") else MID_RGBA
            r = radius if ("hand" in name or "foot" in name) else 0.6 * radius
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, 0, 0]), p, np.eye(3).ravel(), np.array(rgba, np.float32))
            scene.ngeom += 1
        if self.locomotion:                                 # where the reference root is: a green dot + heading
            ref = self._ref(self.t)
            x, y, yaw = ref["xy"][0], ref["xy"][1], ref["yaw"]
            for p, rad in (((x, y, 0.01), 0.05), ((x + 0.14 * np.cos(yaw), y + 0.14 * np.sin(yaw), 0.01), 0.025)):
                if scene.ngeom >= scene.maxgeom:
                    break
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([rad, 0, 0]), np.array(p, float), np.eye(3).ravel(),
                                    np.array(MID_RGBA, np.float32))
                scene.ngeom += 1

    def render(self):
        self.base.render()
        v = self.base.viewer
        if v is not None:
            with v.lock():
                v.user_scn.ngeom = 0
                self.draw_targets(v.user_scn)
            v.sync()

    def close(self):
        self.base.close()
