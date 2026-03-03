import cv2
import mediapipe as mp
import numpy as np


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

    def _empty_output(self):
        return np.full(
            (self.NUM_LANDMARKS, self.LANDMARK_DIM),
            self.missing_value,
            dtype=np.float32,
        )

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
                output[i] = [lm.x, lm.y, lm.z, lm.visibility]

        return output.flatten() if flatten else output

    def close(self):
        self.pose.close()
    def plot_world_landmarks(self, landmarks, ax):
        """
        3D visualization for world landmarks (meters).
        Safely handles None and corrupted inputs.
        """
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
        points = landmarks[valid_mask]

        if points.size == 0:
            return ax

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        ax.scatter(x, y, z, c="green")

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

        return ax

if __name__=="__main__":
    import cv2
    import matplotlib.pyplot as plt 
    extractor = PoseExtractor(missing_value=-1.0)

    cap = cv2.VideoCapture(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extractor.process(frame)
        print(landmarks.shape)
        plt.cla()
        ax = extractor.plot_world_landmarks(landmarks, ax)
        plt.pause(0.05)
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()