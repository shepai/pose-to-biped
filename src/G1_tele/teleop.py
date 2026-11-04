"""
src/G1_tele/teleop.py  -  a person in front of a camera -> the simulated G1 copying them

Teleop ties together the pieces, without any camera, MediaPipe, display or Stable-Baselines3 code
(examples/RL/live_pose.py adds those):

    pose landmarks --HumanToRobot--> 12 robot keypoints --KeypointRetargeter (IK)--> reference pose
    reference pose --LivePoseTrackingEnv + trained policy--> simulated robot

It is a small state machine:

    WAITING   nobody found yet: the robot stands.
    TRACKING  a torso has been visible for `acquire_frames` camera frames in a row: the camera is
              levelled from it, the robot is reset into the person's pose once, and then follows.
    HOLDING   the person was lost: the robot holds the last pose. If they come back within
              `hold_seconds` it carries on; otherwise it eases back to standing (WAITING), and the
              next person found causes a fresh reset.
    A fall resets the robot into the person's current pose (or standing if nobody is there).

    teleop = Teleop(LivePoseTrackingEnv(), policy)      # policy: obs -> action
    for every camera frame:      teleop.on_pose(world_landmarks, visibility, t)  # None if no person
    50 times a second:           info = teleop.step()
"""
import numpy as np

from G1_tele.motion.human import HumanToRobot, KeypointFilter
from G1_tele.motion.kinematics import G1Kinematics
from G1_tele.motion.retarget import KeypointRetargeter

WAITING, TRACKING, HOLDING = "waiting", "tracking", "holding"


class Teleop:
    def __init__(self, env, policy, min_visibility=0.5, acquire_frames=8, hold_seconds=2.0, tau=0.08,
                 return_tau=0.6, level=True, com_weight=0.0, kin=None, one_euro=True):
        self.env, self.policy = env, policy
        self.kin = kin or G1Kinematics()                       # for IK (sharing env.kin is fine: every use runs FK first)
        self.human = HumanToRobot(self.kin, min_visibility)
        self.ik_kw = dict(com_weight=com_weight)
        self.retarget = KeypointRetargeter(self.kin, **self.ik_kw)
        # One-Euro filter on the robot keypoints (True = defaults, a dict = KeypointFilter settings, False = off)
        self.filter = KeypointFilter(**(one_euro if isinstance(one_euro, dict) else {})) if one_euro else None
        self.acquire_frames, self.hold_seconds = int(acquire_frames), float(hold_seconds)
        self.tau, self.return_tau, self.level = float(tau), float(return_tau), bool(level)
        self.state = WAITING
        self.seen, self.recent = 0, []                         # torso frames in a row, their landmarks
        self.pose = None                                       # last (world, visibility) with a torso
        self.target = (env.standing_root, env.standing_joints)
        self.keypoints, self.retarget_err = None, np.nan
        self.lost_at, self.t_pose = None, 0.0
        self.resets = self.falls = 0
        self.last_info = {}
        self.obs, _ = env.reset_to()

    # ------------------------------------------------------------------ camera side
    def on_pose(self, world, visibility, t=None):
        """Call once per camera frame with MediaPipe world landmarks (33, 3) and visibility (33,),
        or with None, None when no person was found. t: the frame's time in seconds (for the
        keypoint filter); defaults to the simulation time."""
        self.t_pose = self.env.t if t is None else float(t)
        if world is None or visibility is None or not self.human.torso_visible(visibility):
            self.seen, self.recent = 0, []
            if self.state == TRACKING:
                self.state, self.lost_at = HOLDING, self.env.t
            return
        world, visibility = np.asarray(world, float), np.asarray(visibility, float)
        self.pose = (world, visibility)
        self.seen += 1
        self.recent = (self.recent + [world])[-self.acquire_frames:]
        if self.state == WAITING:
            if self.seen >= self.acquire_frames:
                self.reset_to_person(calibrate=True)
            return
        if self.state == HOLDING:                              # back within hold_seconds: carry on
            self.state = TRACKING
        self.target = self._retarget(world, visibility)

    def _retarget(self, world, visibility):
        self.keypoints = self.human(world, visibility)
        if self.filter is not None:
            self.keypoints = self.filter(self.keypoints, self.t_pose)
        root, joints, self.retarget_err = self.retarget.step(self.keypoints)
        return root, joints

    def reset_to_person(self, calibrate=False):
        """Put the robot into the current person's pose now (standing if nobody is visible)."""
        if self.pose is None or self.seen == 0:
            self.state, self.target = WAITING, (self.env.standing_root, self.env.standing_joints)
            self.obs, _ = self.env.reset_to()
            return False
        if calibrate and self.level and self.recent:
            self.human.calibrate(np.array(self.recent))
        self.human.reset_contacts()
        self.retarget = KeypointRetargeter(self.kin, **self.ik_kw)          # fresh IK, no warm start
        if self.filter is not None:
            self.filter.reset()
        self.target = self._retarget(*self.pose)
        self.obs, _ = self.env.reset_to(*self.target)
        self.state, self.lost_at = TRACKING, None
        self.resets += 1
        return True

    def recalibrate(self):
        """Re-level the camera from the person's current torso (they should be upright)."""
        if self.recent:
            return self.human.calibrate(np.array(self.recent))
        return None

    # ------------------------------------------------------------------ control side
    def step(self):
        """One control step (env.dt = 20 ms). Returns the env's info dict plus 'state'."""
        env = self.env
        tau = self.tau
        if self.state == HOLDING and env.t - self.lost_at > self.hold_seconds:
            self.state = WAITING                               # gone for good: ease back to standing
            self.target = (env.standing_root, env.standing_joints)
        if self.state == WAITING:
            tau = self.return_tau
        env.set_target(*self.target, tau=tau)
        self.obs, _, terminated, truncated, info = env.step(self.policy(self.obs))
        if terminated or truncated:
            self.falls += int(bool(info.get("fell")))
            if self.state == TRACKING and self.seen > 0:
                self.reset_to_person()
            else:
                self.state, self.target = WAITING, (env.standing_root, env.standing_joints)
                self.obs, _ = env.reset_to()
        info["state"] = self.state
        self.last_info = info
        return info
