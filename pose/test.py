import cv2
from __init__ import PoseExtractor, PARENTS
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