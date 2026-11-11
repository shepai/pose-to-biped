"""
src/G1_tele/envs/live.py

The pose-tracking env with a live reference instead of the motion library, so a policy trained on
PoseTrackingEnv can follow poses that arrive one at a time (from a camera, a network, a script...).

    env = LivePoseTrackingEnv()
    obs, _ = env.reset_to(root, joints)         # robot placed in this pose (e.g. a person's, retargeted)
    while True:
        env.set_target(root, joints)            # newest reference pose, every control step (50 Hz)
        obs, reward, terminated, truncated, info = env.step(policy(obs))

With gym.make("G1-pose-live"), start episodes with env.reset(options={"root": root, "joints": joints})
and call env.unwrapped.set_target(...). A plain reset() restarts at the current reference pose
(standing until a target has been set).

Observations, actions and rewards are exactly those of PoseTrackingEnv, so trained models and their
VecNormalize statistics work unchanged. Differences:
  * set_target() moves the reference smoothly towards the newest pose (time constant tau), with
    joint speeds limited to 3 rad/s like the synthetic training motions (5 rad/s when walking), and
    supplies the joint velocities the policy observes.
  * Episodes never time out, and only a fall ends one: max_kp_err defaults to 1 m, so a pose the
    robot cannot reach is followed as well as possible instead of ending the episode.
  * No start-pose noise (init_noise=0).

Walking (locomotion=True, for models from train_locomotion.py)
  * The root targets you pass are in your own "room" frame (e.g. where the person is relative to
    where they were first seen). The reference root moves towards them with its own, slower time
    constant (root_tau) and is limited to max_root_speed and max_yaw_rate, about the fastest the
    training clips go; the resulting root velocity is what the policy observes and follows.
  * Drifting off course never ends an episode here. Instead, when the robot falls more than
    leash = (metres, radians) behind the reference root, the room frame is shifted (re-anchored) so the
    reference stays within that distance of the robot: the policy never sees an error bigger than
    in training, and the person's later movements are still copied from wherever the robot is.
"""
import numpy as np

from G1_tele.robots import get_robot
from G1_tele.envs.pose_tracking import PoseTrackingEnv
from G1_tele.motion.kinematics import Kinematics, quat_mul, quat_to_mat, rot_z, slerp, yaw_of_quat


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _quat_yaw(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _split_yaw(q):
    """q = yaw rotation * tilt: returns (yaw, tilt quaternion)."""
    yaw = yaw_of_quat(q)
    return yaw, quat_mul(_quat_yaw(-yaw), np.asarray(q, float))


class LiveReference:
    """Stands in for MotionLibrary inside PoseTrackingEnv: one endless 'clip' whose frame is always
    the current reference pose."""

    n_clips = 1

    def __init__(self, root, joints, max_joint_speed=3.0, locomotion=False, root_tau=0.25, max_root_speed=0.6,
                 max_yaw_rate=1.0):
        self.kinds = np.array(["live"])
        self.max_joint_speed = float(max_joint_speed)
        self.locomotion = bool(locomotion)
        self.root_tau, self.max_root_speed, self.max_yaw_rate = float(root_tau), float(max_root_speed), float(max_yaw_rate)
        self.snap(root, joints)

    def snap(self, root, joints):
        """Jump to a pose and stand still (used on reset)."""
        self.root = np.array(root, float)
        self.joints = np.array(joints, float)
        self.jvel = np.zeros_like(self.joints)
        self.lin, self.ang = np.zeros(3), np.zeros(3)

    def update(self, root, joints, dt, tau=0.08):
        """Move part of the way towards a new pose: a first-order filter with time constant tau (s)."""
        a = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
        step = a * (np.asarray(joints, float) - self.joints)
        step = np.clip(step, -self.max_joint_speed * dt, self.max_joint_speed * dt)
        self.joints = self.joints + step
        self.jvel = step / dt
        root = np.asarray(root, float)
        if not self.locomotion:
            self.root = np.concatenate([self.root[:3] + a * (root[:3] - self.root[:3]),
                                        slerp(self.root[3:7], root[3:7], a)])
            return
        # walking: height and tilt as before; position on the floor and heading slower and speed-limited
        old = self.root.copy()
        ar = 1.0 - np.exp(-dt / self.root_tau) if self.root_tau > 0 else 1.0
        dxy = ar * (root[:2] - old[:2])
        n = np.linalg.norm(dxy)
        if n > self.max_root_speed * dt:
            dxy *= self.max_root_speed * dt / n
        yaw0, tilt0 = _split_yaw(old[3:7])
        yaw1, tilt1 = _split_yaw(root[3:7])
        dyaw = np.clip(ar * _wrap(yaw1 - yaw0), -self.max_yaw_rate * dt, self.max_yaw_rate * dt)
        q = quat_mul(_quat_yaw(yaw0 + dyaw), slerp(tilt0, tilt1, a))
        z = old[2] + a * (root[2] - old[2])
        self.root = np.concatenate([old[:2] + dxy, [z], q])
        self.lin = (self.root[:3] - old[:3]) / dt
        w0, v0 = old[3], -old[4:7]                         # body-frame angular velocity, as MotionLibrary
        w1, v1 = q[0], q[1:]
        dw, dv = w0 * w1 - v0 @ v1, w0 * v1 + w1 * v0 + np.cross(v0, v1)
        if dw < 0:
            dv = -dv
        self.ang = 2.0 * dv / dt

    def shift(self, dxy=(0.0, 0.0), dyaw=0.0):
        """Move the reference rigidly: rotate by dyaw about its own position, then translate by dxy."""
        yaw, tilt = _split_yaw(self.root[3:7])
        self.root[3:7] = quat_mul(_quat_yaw(yaw + dyaw), tilt)
        self.root[:2] += np.asarray(dxy, float)
        R = rot_z(dyaw)
        self.lin = R @ self.lin

    # ---- the part of the MotionLibrary interface PoseTrackingEnv uses
    def duration(self, c):
        return 1e9

    def sample_clip(self, rng, kind_weights=None, min_seconds=0.0):
        return 0

    def frame(self, c, t):
        return self.root.copy(), self.joints.copy()

    def velocity(self, c, t):
        """(root linear (world), root angular (body), joint) velocity of the live reference."""
        return self.lin.copy(), self.ang.copy(), self.jvel.copy()


class LivePoseTrackingEnv(PoseTrackingEnv):
    def __init__(self, max_kp_err=1.0, init_noise=0.0, max_joint_speed=None, locomotion=False, root_tau=0.25,
                 max_root_speed=None, max_yaw_rate=1.0, leash=None, **kwargs):
        # joint speed limit of the reference: like the training motions (3 rad/s; walking clips peak at ~5)
        max_joint_speed = max_joint_speed or (5.0 if locomotion else 3.0)
        kwargs.pop("episode_seconds", None)
        kwargs.pop("motion_file", None)                    # the reference is live: no motion library needed
        spec = get_robot(kwargs.get("robot"))
        if kwargs.get("xml_path") is not None:
            spec = spec.with_xml(kwargs["xml_path"])
        k = Kinematics(spec)
        q0 = k.default_qpos()
        q0[2] -= k.keypoints_local(["left_foot"], q0)[0, 2] - k.ankle_height     # standing, feet on the floor
        standing_root, standing_joints = q0[:7].copy(), q0[k.q_adr].copy()
        sc = k.scale                                       # G1 numbers -> this robot's size
        max_root_speed = 0.6 * sc if max_root_speed is None else max_root_speed
        leash = (0.35 * sc, 0.5) if leash is None else leash
        lib = LiveReference(standing_root, standing_joints, max_joint_speed, locomotion, root_tau,
                            max_root_speed, max_yaw_rate)
        super().__init__(motion_file=lib, max_kp_err=max_kp_err, init_noise=init_noise, episode_seconds=1e9,
                         locomotion=locomotion, **kwargs)
        self.standing_root, self.standing_joints = standing_root, standing_joints
        self.end_off_course = False
        self.leash = (float(leash[0]), float(leash[1]))
        self.offset = np.zeros(3)                          # room frame -> world: (x, y, yaw)
        self.clip_kinds = ["live"]
        self.reset()

    # ---- room frame <-> world (only matters with locomotion)
    def to_world(self, root):
        root = np.array(root, float)
        if not self.locomotion:
            return root
        ox, oy, oyaw = self.offset
        root[:2] = rot_z(oyaw)[:2, :2] @ root[:2] + np.array([ox, oy])
        root[3:7] = quat_mul(_quat_yaw(oyaw), root[3:7])
        return root

    def reset(self, seed=None, options=None):
        """options may hold "root" (7,) and "joints" (J,): the pose to start in (root in the room
        frame), and "new_room": True to forget any re-anchoring (a new person)."""
        options = dict(options or {})
        if options.pop("new_room", False):
            self.offset = np.zeros(3)
        if "root" in options and isinstance(self.lib, LiveReference):
            self.lib.snap(self.to_world(options.pop("root")), options.pop("joints"))
        options.setdefault("clip", 0)
        options.setdefault("t0", 0.0)
        return super().reset(seed=seed, options=options)

    def reset_to(self, root=None, joints=None, seed=None, new_room=False):
        """Start a new episode with the robot (and the reference) in this pose; standing if None."""
        if root is None:
            root, joints = self.standing_root, self.standing_joints
        return self.reset(seed=seed, options={"root": root, "joints": joints, "new_room": new_room})

    def set_target(self, root, joints, tau=0.08):
        """Newest reference pose (root in the room frame); call once per control step before step()."""
        self.lib.update(self.to_world(root), joints, self.dt, tau)
        if self.locomotion:
            self._keep_on_leash()

    def _keep_on_leash(self):
        d, ref = self.base.d, self.lib.root
        e = ref[:2] - d.qpos[:2]
        dist = np.linalg.norm(e)
        dyaw = _wrap(yaw_of_quat(ref[3:7]) - yaw_of_quat(d.qpos[3:7]))
        if abs(dyaw) > self.leash[1]:                      # turn the room about the reference root
            r = -(abs(dyaw) - self.leash[1]) * np.sign(dyaw)
            p = ref[:2].copy()
            o = self.offset
            o[:2] = rot_z(r)[:2, :2] @ (o[:2] - p) + p
            o[2] += r
            self.lib.shift(dyaw=r)
        if dist > self.leash[0]:                           # pull the room towards the robot
            dxy = -e * (dist - self.leash[0]) / dist
            self.offset[:2] += dxy
            self.lib.shift(dxy=dxy)

    @property
    def reference(self):
        """Current reference (root (7,) in the world, joints (J,))."""
        return self.lib.root.copy(), self.lib.joints.copy()
