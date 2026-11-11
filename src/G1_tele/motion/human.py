"""
MediaPipe Pose -> robot keypoints, for driving the pose-tracking policy from a camera.

MediaPipe's pose_world_landmarks are 33 points in metres with the origin between the hips and the
image's axes: x right, y down, z away from the camera. HumanToRobot turns one such pose into the
12 keypoints the policy follows ("full" set: hands, elbows, shoulders, hips, knees, feet), in the
heading frame of G1_tele.motion.kinematics (x forward, y left, z up, origin = pelvis on the floor)
and at the robot's size:

  * Levelling: a webcam is rarely level, which makes an upright person look tilted. calibrate()
    takes "up" from the person's torso while they stand or sit upright (at most 30 degrees off).
  * Heading: forward is worked out from the hips (or the shoulders if the hips are unclear), so the
    person can stand at an angle to the camera. Turning on the spot is not copied: the policy
    tracks body pose relative to the pelvis, not where the robot faces.
  * Proportions: every limb keeps the person's direction but gets the robot's length, so any
    adult maps onto the robot (the 1.3 m G1, the 1.8 m H1, ...) with reachable targets.
  * Missing limbs: an arm or leg whose landmarks aren't visible enough (legs hidden by a desk, an
    arm out of frame) is put in the robot's default pose, carried by the torso (arms) or pelvis (legs).
  * Ground: the pose is moved up or down so the lower foot stands on the floor, and a foot within a
    few centimetres of the floor counts as planted and is put exactly on it (with hysteresis), so
    landmark noise doesn't make a planted foot look lifted to the policy.

Walking (for policies trained with locomotion=True): the world landmarks are centred on the hips, so
they don't say where the person is. hip_position() recovers that from the image landmarks: it finds
the shift T that makes the metric skeleton (world landmarks + T) project onto the image landmarks
through a pinhole camera with the given field of view - a small linear least-squares problem.
room_pose() turns it, with the heading, into a position on the floor and a heading angle in the
levelled camera frame. Depth is the least certain part (it comes from the person's apparent size),
so it is smoothed more than the sideways position by the caller (Teleop).

Pure numpy: MediaPipe itself is only needed by whatever produces the landmarks
(examples/RL/live_pose.py).
"""
import numpy as np

from G1_tele.motion.kinematics import KEYPOINT_SETS, Kinematics

# MediaPipe Pose landmark indices used here
LM = dict(left_shoulder=11, right_shoulder=12, left_elbow=13, right_elbow=14, left_wrist=15, right_wrist=16,
          left_pinky=17, right_pinky=18, left_index=19, right_index=20, left_hip=23, right_hip=24,
          left_knee=25, right_knee=26, left_ankle=27, right_ankle=28)

# MediaPipe axes (x right, y down, z away from the camera) -> (x right, y away from the camera, z up).
# A proper rotation (det = +1), so left and right are kept.
MP_TO_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
MAX_LEVEL_DEG = 30.0


def _unit(v, fallback=None):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.asarray(fallback, float) if fallback is not None else np.zeros(3)
    return v / n


def _frame(up, left):
    """Right-handed frame, columns (forward, left, up), from an up vector and a rough left vector."""
    z = _unit(np.asarray(up, float), [0, 0, 1])
    left = np.asarray(left, float)
    y = _unit(left - np.dot(left, z) * z, [0, 1, 0])
    return np.stack([np.cross(y, z), y, z], axis=1)


def hip_position(world, image, visibility, aspect, hfov_deg=65.0, min_visibility=0.5):
    """Hip centre relative to the camera (metres, MediaPipe axes: x right, y down, z away).

    world: (33, 3) pose_world_landmarks (metric, hip-centred); image: (33, 2) pose_landmarks x, y
    (0-1 across the image); aspect: image width / height; hfov_deg: the camera's horizontal field of
    view. Each visible landmark i must satisfy  (X_i + T) projected = image_i, i.e.
        Tx - a_i Tz = a_i Z_i - X_i   and   Ty - b_i Tz = b_i Z_i - Y_i
    with (a_i, b_i) its normalised image coordinates. Returns None if too few landmarks are visible."""
    W, I, v = np.asarray(world, float), np.asarray(image, float), np.asarray(visibility, float)
    use = v >= min_visibility
    if use.sum() < 6:
        return None
    t = np.tan(np.radians(hfov_deg) / 2)
    a = (I[use, 0] - 0.5) * 2 * t
    b = (I[use, 1] - 0.5) * 2 * t / aspect
    X, Y, Z = W[use, 0], W[use, 1], W[use, 2]
    n = len(a)
    A = np.zeros((2 * n, 3))
    A[:n, 0], A[:n, 2], A[n:, 1], A[n:, 2] = 1.0, -a, 1.0, -b
    y = np.concatenate([a * Z - X, b * Z - Y])
    w = np.sqrt(np.concatenate([v[use], v[use]]))
    T, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    return T if T[2] > 0.3 else None                     # in front of the camera


def place_in_room(root, room):
    """A root pose (7,) from the person's heading frame (pelvis near the origin, facing +x) placed at
    room pose (x, y, yaw): the robot's root where the person stands and faces."""
    root = np.array(root, float)
    x, y, yaw = room
    c, s = np.cos(yaw), np.sin(yaw)
    root[:2] = np.array([x + c * root[0] - s * root[1], y + s * root[0] + c * root[1]])
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    w1, v1 = qz[0], qz[1:]
    w2, v2 = root[3], root[4:7]
    root[3:7] = np.concatenate([[w1 * w2 - v1 @ v2], w1 * v2 + w2 * v1 + np.cross(v1, v2)])
    return root


def _rotation_between(a, b, max_deg=180.0):
    """Smallest rotation taking direction a to direction b, limited to max_deg (Rodrigues)."""
    a, b = _unit(a, [0, 0, 1]), _unit(b, [0, 0, 1])
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-9:
        return np.eye(3)
    ang = min(np.arctan2(s, c), np.radians(max_deg))
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class HumanToRobot:
    """converter = HumanToRobot(); keypoints = converter(world_landmarks, visibility)
    (HumanToRobot(robot="h1") or HumanToRobot(kin) for another robot)

    world_landmarks: (33, 3) MediaPipe pose_world_landmarks; visibility: (33,) in [0, 1].
    Returns dict name -> (3,) heading-frame position for the 12 "full" keypoints. After each call,
    converter.tracked says which limbs came from the person (the rest are the default pose).
    """

    def __init__(self, kin=None, min_visibility=0.5, plant_height=0.03, lift_height=0.05, robot=None):
        k = kin or Kinematics(robot)
        self.k = k
        self.ankle_height = k.ankle_height
        self.names = list(KEYPOINT_SETS["full"])
        q0 = k.default_qpos()
        q0[2] -= k.keypoints_local(["left_foot"], q0)[0, 2] - k.ankle_height      # feet on the floor
        self.standing_qpos = q0
        self.standing_root, self.standing_joints = q0[:7].copy(), q0[k.q_adr].copy()
        P = dict(zip(self.names, k.keypoints_local(self.names, q0)))
        self.P0 = P
        self.hip0 = (P["left_hip"] + P["right_hip"]) / 2
        sh0 = (P["left_shoulder"] + P["right_shoulder"]) / 2
        self.torso0 = _frame(sh0 - self.hip0, P["left_shoulder"] - P["right_shoulder"])
        self.pelvis0 = _frame([0, 0, 1], P["left_hip"] - P["right_hip"])

        def length(a, b):
            return float(np.mean([np.linalg.norm(P[f"{s}_{b}"] - P[f"{s}_{a}"]) for s in ("left", "right")]))
        self.length = {"upper_arm": length("shoulder", "elbow"), "forearm": length("elbow", "hand"),
                       "thigh": length("hip", "knee"), "shin": length("knee", "foot")}
        self.min_visibility = float(min_visibility)
        self.plant_height, self.lift_height = float(plant_height) * k.scale, float(lift_height) * k.scale
        self.R_level = np.eye(3)
        self.tracked = {}
        self.planted = {"left_foot": True, "right_foot": True}
        self.heading = 0.0

    # ------------------------------------------------------------------ checks and calibration
    def torso_visible(self, visibility):
        """Both shoulders clearly visible and both hips at least roughly located."""
        v = np.asarray(visibility, float)
        return bool(min(v[11], v[12]) >= self.min_visibility and min(v[23], v[24]) >= 0.5 * self.min_visibility)

    def calibrate(self, world_landmarks):
        """Level the camera from the torso of a person standing or sitting upright.
        world_landmarks: (33, 3) or several frames (N, 33, 3). Returns the correction in degrees."""
        W = np.asarray(world_landmarks, float).reshape(-1, 33, 3) @ MP_TO_UP.T
        up = np.mean((W[:, 11] + W[:, 12]) / 2 - (W[:, 23] + W[:, 24]) / 2, axis=0)
        self.R_level = _rotation_between(up, [0.0, 0.0, 1.0], MAX_LEVEL_DEG)
        return float(np.degrees(np.arccos(np.clip(self.R_level[2, 2], -1, 1))))

    def reset_calibration(self):
        self.R_level = np.eye(3)

    def reset_contacts(self):
        """Forget which feet were planted (call when tracking a new person)."""
        self.planted = {"left_foot": True, "right_foot": True}

    # ------------------------------------------------------------------ conversion
    def levelled(self, world_landmarks):
        """Landmarks in the levelled camera frame (x right, y away from the camera, z up)."""
        return (np.asarray(world_landmarks, float) @ MP_TO_UP.T) @ self.R_level.T

    def __call__(self, world_landmarks, visibility, left=None):
        """left: optional direction of the person's left in the levelled camera frame (e.g. smoothed
        over time); by default it is taken from the hips in this frame."""
        W = self.levelled(world_landmarks)
        v = np.asarray(visibility, float)
        P0, L = self.P0, self.length

        def lm(name):
            return W[LM[name]]

        def seen(*names):
            return all(v[LM[n]] >= self.min_visibility for n in names)

        hip_mid = (lm("left_hip") + lm("right_hip")) / 2
        if left is None:
            left = (lm("left_hip") - lm("right_hip")) if seen("left_hip", "right_hip") else \
                (lm("left_shoulder") - lm("right_shoulder"))
        left = np.array([left[0], left[1], 0.0])
        H = _frame([0, 0, 1], left)                          # the person's heading frame (camera levelled)
        self.heading = float(np.arctan2(H[1, 0], H[0, 0]))   # where they face, in the levelled camera frame

        def h(p):                                            # -> heading frame, origin between the hips
            return H.T @ (p - hip_mid)

        sh_mid = (lm("left_shoulder") + lm("right_shoulder")) / 2
        R_torso = _frame(h(sh_mid), h(lm("left_shoulder")) - h(lm("right_shoulder"))) @ self.torso0.T
        R_pelvis = _frame([0, 0, 1], h(lm("left_hip")) - h(lm("right_hip"))) @ self.pelvis0.T

        out, self.tracked = {}, {}
        for s in ("left", "right"):
            hip = self.hip0 + R_pelvis @ (P0[f"{s}_hip"] - self.hip0)
            sho = self.hip0 + R_torso @ (P0[f"{s}_shoulder"] - self.hip0)
            out[f"{s}_hip"], out[f"{s}_shoulder"] = hip, sho

            # arm: shoulder -> elbow -> palm centre (the robot's "hand" keypoint)
            if seen(f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"):
                hand = lm(f"{s}_wrist")
                if seen(f"{s}_index", f"{s}_pinky"):
                    hand = (lm(f"{s}_wrist") + lm(f"{s}_index") + lm(f"{s}_pinky")) / 3
                d_up = R_torso @ _unit(P0[f"{s}_elbow"] - P0[f"{s}_shoulder"])
                elbow = sho + L["upper_arm"] * _unit(h(lm(f"{s}_elbow")) - h(lm(f"{s}_shoulder")), d_up)
                out[f"{s}_hand"] = elbow + L["forearm"] * _unit(h(hand) - h(lm(f"{s}_elbow")), d_up)
                out[f"{s}_elbow"] = elbow
                self.tracked[f"{s}_arm"] = True
            else:
                out[f"{s}_elbow"] = sho + R_torso @ (P0[f"{s}_elbow"] - P0[f"{s}_shoulder"])
                out[f"{s}_hand"] = sho + R_torso @ (P0[f"{s}_hand"] - P0[f"{s}_shoulder"])
                self.tracked[f"{s}_arm"] = False

            # leg: hip -> knee -> ankle (the robot's "foot" keypoint)
            if seen(f"{s}_hip", f"{s}_knee", f"{s}_ankle"):
                down = np.array([0.0, 0.0, -1.0])
                knee = hip + L["thigh"] * _unit(h(lm(f"{s}_knee")) - h(lm(f"{s}_hip")), down)
                out[f"{s}_foot"] = knee + L["shin"] * _unit(h(lm(f"{s}_ankle")) - h(lm(f"{s}_knee")), down)
                out[f"{s}_knee"] = knee
                self.tracked[f"{s}_leg"] = True
            else:
                out[f"{s}_knee"] = hip + (P0[f"{s}_knee"] - P0[f"{s}_hip"])
                out[f"{s}_foot"] = hip + (P0[f"{s}_foot"] - P0[f"{s}_hip"])
                self.tracked[f"{s}_leg"] = False

        A = self.ankle_height
        dz = A - min(out["left_foot"][2], out["right_foot"][2])      # lower foot on the floor
        out = {n: out[n] + np.array([0.0, 0.0, dz]) for n in self.names}
        for f in ("left_foot", "right_foot"):                  # planted feet exactly on the floor
            lift = out[f][2] - A
            self.planted[f] = lift < (self.lift_height if self.planted[f] else self.plant_height)
            if self.planted[f]:
                out[f][2] = A
        return out

    def room_pose(self, world_landmarks, image_landmarks, visibility, aspect, hfov_deg=65.0):
        """(x, y) of the hip centre on the floor and the heading (rad), both in the levelled camera frame
        (x right, y away from the camera), or None if the position can't be worked out.
        Call after __call__() for the same frame (it uses the heading worked out there)."""
        T = hip_position(world_landmarks, image_landmarks, visibility, aspect, hfov_deg, self.min_visibility)
        if T is None:
            return None
        p = self.levelled(T[None])[0]
        return np.array([p[0], p[1]]), self.heading


class KeypointFilter:
    """One-Euro filter (Casiez et al., 2012) for a dict of 3-D keypoints: strong smoothing while a
    point is still (removes landmark jitter), little while it moves fast (keeps lag low).

        f = KeypointFilter(); smooth = f(keypoints, t)        # t in seconds

    min_cutoff (Hz): smoothing at rest, lower = smoother. beta (1/m): how quickly smoothing relaxes
    as a point speeds up. d_cutoff (Hz): smoothing of the speed estimate itself.
    """

    def __init__(self, min_cutoff=1.5, beta=2.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.reset()

    def reset(self):
        self.x, self.dx, self.t = None, None, None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))

    def __call__(self, keypoints, t):
        names = list(keypoints)
        x = np.array([keypoints[n] for n in names], float)
        if self.x is None or self.t is None or t <= self.t or self.x.shape != x.shape:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return dict(zip(names, x))
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx, axis=1, keepdims=True)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return dict(zip(names, self.x.copy()))
