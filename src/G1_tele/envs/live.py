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
    joint speeds limited to 3 rad/s like the synthetic training motions, and supplies the joint
    velocities the policy observes.
  * Episodes never time out, and only a fall ends one: max_kp_err defaults to 1 m, so a pose the
    robot cannot reach is followed as well as possible instead of ending the episode.
  * No start-pose noise (init_noise=0).
"""
import numpy as np

from G1_tele.envs.pose_tracking import PoseTrackingEnv
from G1_tele.motion.kinematics import ANKLE_HEIGHT, slerp


class LiveReference:
    """Stands in for MotionLibrary inside PoseTrackingEnv: one endless 'clip' whose frame is always
    the current reference pose."""

    n_clips = 1

    def __init__(self, root, joints, max_joint_speed=3.0):
        self.kinds = np.array(["live"])
        self.max_joint_speed = float(max_joint_speed)
        self.snap(root, joints)

    def snap(self, root, joints):
        """Jump to a pose and stand still (used on reset)."""
        self.root = np.array(root, float)
        self.joints = np.array(joints, float)
        self.jvel = np.zeros_like(self.joints)

    def update(self, root, joints, dt, tau=0.08):
        """Move part of the way towards a new pose: a first-order filter with time constant tau (s)."""
        a = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
        step = a * (np.asarray(joints, float) - self.joints)
        step = np.clip(step, -self.max_joint_speed * dt, self.max_joint_speed * dt)
        self.joints = self.joints + step
        self.jvel = step / dt
        root = np.asarray(root, float)
        self.root = np.concatenate([self.root[:3] + a * (root[:3] - self.root[:3]), slerp(self.root[3:7], root[3:7], a)])

    # ---- the part of the MotionLibrary interface PoseTrackingEnv uses
    def duration(self, c):
        return 1e9

    def sample_clip(self, rng, kind_weights=None, min_seconds=0.0):
        return 0

    def frame(self, c, t):
        return self.root.copy(), self.joints.copy()

    def velocity(self, c, t):
        """(root linear, root angular, joint) velocity. Root velocities are only used to start an
        episode, which always starts at rest here."""
        return np.zeros(3), np.zeros(3), self.jvel.copy()


class LivePoseTrackingEnv(PoseTrackingEnv):
    def __init__(self, max_kp_err=1.0, init_noise=0.0, max_joint_speed=3.0, **kwargs):
        kwargs.pop("episode_seconds", None)
        super().__init__(max_kp_err=max_kp_err, init_noise=init_noise, episode_seconds=1e9, **kwargs)
        k = self.kin
        q0 = k.default_qpos()
        q0[2] -= k.keypoints_local(["left_foot"], q0)[0, 2] - ANKLE_HEIGHT     # standing, feet on the floor
        self.standing_root, self.standing_joints = q0[:7].copy(), q0[k.q_adr].copy()
        self.lib = LiveReference(self.standing_root, self.standing_joints, max_joint_speed)
        self.clip_kinds = ["live"]
        self.reset()

    def reset(self, seed=None, options=None):
        """options may hold "root" (7,) and "joints" (29,): the pose to start in."""
        options = dict(options or {})
        if "root" in options and isinstance(self.lib, LiveReference):
            self.lib.snap(options.pop("root"), options.pop("joints"))
        options.setdefault("clip", 0)
        options.setdefault("t0", 0.0)
        return super().reset(seed=seed, options=options)

    def reset_to(self, root=None, joints=None, seed=None):
        """Start a new episode with the robot (and the reference) in this pose; standing if None."""
        if root is None:
            root, joints = self.standing_root, self.standing_joints
        return self.reset(seed=seed, options={"root": root, "joints": joints})

    def set_target(self, root, joints, tau=0.08):
        """Newest reference pose; call once per control step before step()."""
        self.lib.update(root, joints, self.dt, tau)

    @property
    def reference(self):
        """Current reference (root (7,), joints (29,))."""
        return self.lib.root.copy(), self.lib.joints.copy()
