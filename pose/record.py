import cv2
import matplotlib.pyplot as plt
import numpy as np
from __init__ import *

# ================= SETUP =================
extractor = PoseExtractor(missing_value=-1.0)
cap = cv2.VideoCapture(0)

# ---- Create Plot First ----
fig = plt.figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(111, projection="3d")

# Force an initial draw to get real plot size
fig.canvas.draw()

plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
plot_w, plot_h = fig.canvas.get_width_height()
plot_img = plot_img.reshape((plot_h, plot_w, 3))
plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

# ---- Get webcam size ----
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---- Video size (side by side) ----
video_width = frame_width + plot_w
video_height = max(frame_height, plot_h)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    "C:/Users/dexte/Documents/GitHub/pose-to-biped/assets/combined_output.avi",
    fourcc,
    20.0,
    (video_width, video_height)
)

print("Recording started... Press ESC to stop.")

# ================= MAIN LOOP =================
c = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extractor.process(frame)
    c += 1

    if c > 2:
        angles = extractor.compute_joint_angle_changes(PARENTS)
        print(angles)

    # ---- Update 3D Plot ----
    plt.cla()
    ax = extractor.plot_world_landmarks(landmarks, ax)

    # Force redraw
    fig.canvas.draw()

    # ---- Capture Plot Image ----
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape((plot_h, plot_w, 3))
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    # ---- Resize Frames (safety) ----
    frame = cv2.resize(frame, (frame_width, frame_height))
    plot_img = cv2.resize(plot_img, (plot_w, plot_h))

    # ---- Combine Side-by-Side ----
    combined = np.zeros((video_height, video_width, 3), dtype=np.uint8)

    combined[0:frame_height, 0:frame_width] = frame
    combined[0:plot_h, frame_width:frame_width + plot_w] = plot_img

    # ---- Write Video ----
    out.write(combined)

    # ---- Live Preview ----
    cv2.imshow("Recording", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# ================= CLEANUP =================
cap.release()
out.release()
extractor.close()
cv2.destroyAllWindows()
plt.close(fig)

print("Video saved as combined_output.mp4")