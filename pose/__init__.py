import cv2
import mediapipe as mp
import numpy as np

PARENTS = {
    # Upper body
    11: 23,   # left shoulder  ← left hip
    13: 11,   # left elbow     ← left shoulder
    15: 13,   # left wrist     ← left elbow

    12: 24,   # right shoulder ← right hip
    14: 12,
    16: 14,

    # Lower body
    25: 23,   # left knee  ← left hip
    27: 25,   # left ankle ← left knee

    26: 24,   # right knee
    28: 26,

    # Spine connection
    23: 24,   # left hip  ← right hip (root link)
}
class PoseExtractor:
    """
    MediaPipe Pose wrapper that always returns a fixed-format vector.
    
    Output format:
        shape (33, 4)
        Each row: [x, y, z, visibility]
    """

    NUM_LANDMARKS = 33
    LANDMARK_DIM = 4

    def __init__(
        self,
        missing_value=np.nan,
        use_world_landmarks=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.missing_value = missing_value
        self.use_world_landmarks = use_world_landmarks

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarks_prev=None
        self.landmarks_curr=None
    def _empty_output(self):
        return np.full(
            (self.NUM_LANDMARKS, self.LANDMARK_DIM),
            self.missing_value,
            dtype=np.float32,
        )
    def to_local_space(self, landmarks, missing_value=-1.0):
        """
        Converts world landmarks to root-centered local coordinates.

        Args:
            landmarks: (33,4) array
        Returns:
            local_landmarks: (33,4) array (root-centered)
            root: (3,) torso midpoint
        """
        if len(landmarks) != 33:
            return None, None
        local_landmarks = landmarks.copy()
        # Get torso points (left and right hip)
        left_hip = landmarks[23, :3]
        right_hip = landmarks[24, :3]
        
        # Check for missing hips
        if (
            np.any(left_hip == missing_value)
            or np.any(right_hip == missing_value)
        ):
            return None, None
        
        # Compute midpoint
        root = (left_hip + right_hip) / 2.0

        # Subtract root from all valid landmarks
        for i in range(33):
            if not np.any(landmarks[i, :3] == missing_value):
                local_landmarks[i, :3] -= root
        local_landmarks[:,0]*=-1
        local_landmarks[:,2]*=-1
        return local_landmarks, root
    def process(self, image_bgr, flatten=False):
        """
        Args:
            image_bgr: np.ndarray (H, W, 3) in BGR format
            flatten: if True → returns shape (132,)
        
        Returns:
            np.ndarray of shape (33, 4) or (132,)
        """

        output = self._empty_output()

        if image_bgr is None:
            return output.flatten() if flatten else output

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if self.use_world_landmarks:
            landmarks = results.pose_world_landmarks
        else:
            landmarks = results.pose_landmarks

        if landmarks is not None:
            for i, lm in enumerate(landmarks.landmark):
                output[i] = [lm.z, lm.x, lm.y, lm.visibility]

        return output.flatten() if flatten else output

    def close(self):
        self.pose.close()
    def plot_world_landmarks(self, landmarks, ax, points=[],scale=True):
        """
        3D visualization for world landmarks (meters).
        Safely handles None and corrupted inputs.
        """
        self.landmarks_prev=self.landmarks_curr
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        if landmarks is None:
            return ax

        try:
            landmarks = np.array(landmarks)
        except Exception:
            return ax
        
        if landmarks.shape[0] != self.NUM_LANDMARKS:
            return ax

        # Extract valid points
        valid_mask = landmarks[:, 0] != self.missing_value
        points_ = landmarks[valid_mask]
        self.landmarks_curr=points_.copy()
        if points_.size == 0:
            return ax

        x = points_[:, 0]
        y = points_[:, 1]
        z = points_[:, 2]

        ax.scatter(x, y, z, c="green")
        for i in range(len(points)):
            point=points[i]
            ax.scatter(point[0], point[1], point[2], c="red", s=30)
        # Draw skeleton connections
        for start, end in self.mp_pose.POSE_CONNECTIONS:
            if (
                start < len(valid_mask)
                and end < len(valid_mask)
                and valid_mask[start]
                and valid_mask[end]
            ):
                xs = [landmarks[start, 0], landmarks[end, 0]]
                ys = [landmarks[start, 1], landmarks[end, 1]]
                zs = [landmarks[start, 2], landmarks[end, 2]]

                ax.plot(xs, ys, zs, c="blue")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("World Pose")
        if scale:
            x_range = landmarks[:,0].max() - landmarks[:,0].min()
            y_range = landmarks[:,1].max() - landmarks[:,1].min()
            z_range = landmarks[:,2].max() - landmarks[:,2].min()
            max_range = max(x_range, y_range, z_range)

            # Compute midpoints
            x_mid = (landmarks[:,0].max() + landmarks[:,0].min()) / 2
            y_mid = (landmarks[:,1].max() + landmarks[:,1].min()) / 2
            z_mid = (landmarks[:,2].max() + landmarks[:,2].min()) / 2

            # Set limits symmetrically around the midpoint
            ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
            ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
            ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
        return ax
    def compute_joint_angle_changes(self, parents):
        landmarks_prev=self.landmarks_prev
        landmarks_curr=self.landmarks_curr
        angle_changes = {}

        for joint, parent in parents.items():
            try:
                # Bone vectors at t
                v_prev = landmarks_prev[joint] - landmarks_prev[parent]
                # Bone vectors at t+1
                v_curr = landmarks_curr[joint] - landmarks_curr[parent]
                # Skip degenerate bones
                if np.linalg.norm(v_prev) < 1e-6 or np.linalg.norm(v_curr) < 1e-6:
                    angle_changes[joint] = 0.0
                    continue
                # Normalize
                v_prev = v_prev / np.linalg.norm(v_prev)
                v_curr = v_curr / np.linalg.norm(v_curr)
                # Angle between them
                dot = np.clip(np.dot(v_prev, v_curr), -1.0, 1.0)
                angle = np.arccos(dot)
            except:
                angle=0
            angle_changes[joint] = angle

        return angle_changes
if __name__=="__main__":
    import cv2
    import matplotlib.pyplot as plt 
    extractor = PoseExtractor(missing_value=-1.0)

    cap = cv2.VideoCapture(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    c=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extractor.process(frame)
        c+=1
        if c>2:
            angles=extractor.compute_joint_angle_changes(PARENTS)
            print(angles)
        plt.cla()
        ax = extractor.plot_world_landmarks(landmarks, ax)
        plt.pause(0.05)
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()