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

Walking (walk=True, with LivePoseTrackingEnv(locomotion=True) and a model from train_locomotion.py):
also pass the image landmarks to on_pose (image=(33, 2) pose_landmarks x, y; aspect = width / height).
Where the person stands and which way they face are then followed too, relative to where they were
when first seen (the "room" frame): walking towards the camera while facing it is walking forwards,
a step to their left is a step to the robot's left. Position comes from HumanToRobot.room_pose() and
is smoothed (more than the pose: depth from one camera is noisy); a sudden jump of the heading by more
than max_heading_jump (MediaPipe swapping left and right, usually when someone turns their back) is
treated as a bad frame and skipped.
"""
import numpy as np

from G1_tele.motion.human import HumanToRobot, KeypointFilter, place_in_room
from G1_tele.motion.kinematics import rot_z
from G1_tele.motion.retarget import KeypointRetargeter


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi



WAITING, TRACKING, HOLDING = "waiting", "tracking", "holding"


class Teleop:
    def __init__(self, env, policy, min_visibility=0.5, acquire_frames=8, hold_seconds=2.0, tau=0.08,
                 return_tau=0.6, level=True, com_weight=0.0, kin=None, one_euro=True, walk=False, hfov=65.0,
                 max_heading_jump=1.0):
        self.env, self.policy = env, policy
        self.walk = bool(walk) and bool(getattr(env, "locomotion", False))
        self.hfov, self.max_heading_jump = float(hfov), float(max_heading_jump)
        self.xy_filter = KeypointFilter(min_cutoff=0.4, beta=0.6)    # room position (m)
        self.yaw_filter = KeypointFilter(min_cutoff=0.6, beta=0.3)   # heading (rad)
        self.room0, self.room, self.heading_prev = None, np.zeros(3), 0.0
        self.image, self.glitches = None, 0
        self.kin = kin or env.kin                              # the env's robot (sharing env.kin is fine: every use runs FK first)
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
    def on_pose(self, world, visibility, t=None, image=None, aspect=None):
        """Call once per camera frame with MediaPipe world landmarks (33, 3) and visibility (33,),
        or with None, None when no person was found. t: the frame's time in seconds (for the
        keypoint filter); defaults to the simulation time. image, aspect: the image landmarks (33, 2)
        and the image's width / height, needed to follow walking (walk=True)."""
        self.t_pose = self.env.t if t is None else float(t)
        self.image = (np.asarray(image, float), float(aspect)) if image is not None and aspect else None
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
        target = self._retarget(world, visibility)
        if target is not None:
            self.target = target

    def _retarget(self, world, visibility):
        keypoints = self.human(world, visibility)
        if self.walk and not self._update_room(world, visibility):
            return None                                        # a bad frame (left and right swapped)
        self.keypoints = keypoints
        if self.filter is not None:
            self.keypoints = self.filter(self.keypoints, self.t_pose)
        root, joints, self.retarget_err = self.retarget.step(self.keypoints)
        if self.walk:
            root = place_in_room(root, self.room)
        return root, joints

    def _update_room(self, world, visibility):
        """Where the person is in the room frame (smoothed). False for a frame to skip."""
        if self.image is None:
            return True                                        # no image landmarks: keep the last position
        r = self.human.room_pose(world, self.image[0], visibility, self.image[1], self.hfov)
        if r is None:
            return True
        p, heading = r
        if self.room0 is None:                                 # first sight: this is the origin
            self.room0, self.heading_prev = (p.copy(), heading), 0.0
            self.xy_filter.reset()
            self.yaw_filter.reset()
        p0, h0 = self.room0
        rel = self.heading_prev + _wrap(_wrap(heading - h0) - self.heading_prev)       # unwrapped
        if abs(rel - self.heading_prev) > self.max_heading_jump:
            self.glitches += 1
            return False
        self.heading_prev = rel
        d = rot_z(-h0)[:2, :2] @ (p - p0)
        xy = self.xy_filter({"p": np.array([d[0], d[1], 0.0])}, self.t_pose)["p"]
        yaw = self.yaw_filter({"h": np.array([rel, 0.0, 0.0])}, self.t_pose)["h"][0]
        self.room = np.array([xy[0], xy[1], yaw])
        return True

    def _standing_here(self):
        """The standing pose, where the person was last seen (walking) or at the origin."""
        root = self.env.standing_root
        return (place_in_room(root, self.room) if self.walk else root), self.env.standing_joints

    def reset_to_person(self, calibrate=False):
        """Put the robot into the current person's pose now (standing if nobody is visible)."""
        if self.pose is None or self.seen == 0:
            self.state, self.target = WAITING, self._standing_here()
            self.obs, _ = self.env.reset_to(*self.target)
            return False
        if calibrate and self.level and self.recent:
            self.human.calibrate(np.array(self.recent))
        if calibrate:                                          # a new person: a new room frame
            self.room0, self.room, self.heading_prev = None, np.zeros(3), 0.0
        self.human.reset_contacts()
        self.retarget = KeypointRetargeter(self.kin, **self.ik_kw)          # fresh IK, no warm start
        if self.filter is not None:
            self.filter.reset()
        target = self._retarget(*self.pose)
        self.target = target if target is not None else self._standing_here()
        self.obs, _ = self.env.reset_to(*self.target, new_room=calibrate)
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
            self.target = self._standing_here()
        if self.state == WAITING:
            tau = self.return_tau
        env.set_target(*self.target, tau=tau)
        self.obs, _, terminated, truncated, info = env.step(self.policy(self.obs))
        if terminated or truncated:
            self.falls += int(bool(info.get("fell")))
            if self.state == TRACKING and self.seen > 0:
                self.reset_to_person()
            else:
                self.state, self.target = WAITING, self._standing_here()
                self.obs, _ = env.reset_to(*self.target)
        info["state"] = self.state
        if self.walk:
            info["room"] = self.room.copy()
        self.last_info = info
        return info
