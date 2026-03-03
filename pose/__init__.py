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
    def draw(self, image_bgr, landmarks, visibility_threshold=0.0):
        """
        Draw only valid landmarks and valid connections.

        Args:
            image_bgr: original image
            landmarks: (33,4) array from process()
            visibility_threshold: only draw points above this visibility
        """

        image = image_bgr.copy()
        h, w, _ = image.shape

        valid_mask = landmarks[:, 0] != self.missing_value

        #Draw points
        for i in range(self.NUM_LANDMARKS):
            if valid_mask[i] and landmarks[i, 3] > visibility_threshold:
                x = int(landmarks[i, 0] * w)
                y = int(landmarks[i, 1] * h)
                print(x,y)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        #Draw skeleton connections
        for start, end in self.mp_pose.POSE_CONNECTIONS:
            if (
                valid_mask[start]
                and valid_mask[end]
                and landmarks[start, 3] > visibility_threshold
                and landmarks[end, 3] > visibility_threshold
            ):
                x1 = int(landmarks[start, 0] * w)
                y1 = int(landmarks[start, 1] * h)
                x2 = int(landmarks[end, 0] * w)
                y2 = int(landmarks[end, 1] * h)

                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return image

if __name__=="__main__":
    import cv2
    extractor = PoseExtractor(missing_value=-1.0)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extractor.process(frame)
        print(landmarks.shape)
        frame_vis = extractor.draw(frame, landmarks, visibility_threshold=0.5)

        cv2.imshow("Webcam", frame_vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()