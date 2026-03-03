import cv2
import mediapipe as mp
import numpy as np

MISSING_VALUE = np.nan   # Could also use -1.0 if you prefer
USE_WORLD_LANDMARKS = False  # True = 3D world coords, False = normalized image coords

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

NUM_LANDMARKS = 33
LANDMARK_DIM = 4  # x, y, z, visibility

def extract_landmarks(results):
    """
    Returns a fixed-size numpy array of shape (33, 4)
    Missing landmarks are filled with MISSING_VALUE.
    """
    output = np.full((NUM_LANDMARKS, LANDMARK_DIM), MISSING_VALUE, dtype=np.float32)

    if USE_WORLD_LANDMARKS:
        landmarks = results.pose_world_landmarks
    else:
        landmarks = results.pose_landmarks

    if landmarks is None:
        return output

    for i, lm in enumerate(landmarks.landmark):
        output[i] = [lm.x, lm.y, lm.z, lm.visibility]

    return output


# ----------------------------
# Webcam Loop
# ----------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    landmark_array = extract_landmarks(results)

    print("Shape:", landmark_array.shape)

    flattened = landmark_array.flatten()  # shape (132,)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose.close()