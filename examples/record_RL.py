import sys
sys.path.append("/home/dexter/Documents/GitHub/pose-to-biped/")

import cv2
import numpy as np
import mujoco
from sim.environment import *
from stable_baselines3 import SAC

# ------------------------
# Environment + Model
# ------------------------
vec_env = robo_gym(
    path_to_scene="/home/dexter/Documents/GitHub/pose-to-biped/Robots/scene.xml",
    path_to_urdf="/home/dexter/Documents/GitHub/pose-to-biped/Robots/h1_with_hand.urdf"
)

vec_env.set_dataset("/home/dexter/Documents/GitHub/pose-to-biped/models/")

model = SAC.load(
    "/home/dexter/Documents/GitHub/pose-to-biped/models/sac/test_SAC_40000_steps.zip",
    device="cpu"
)

obs, _ = vec_env.reset()

# ------------------------
# Video settings
# ------------------------
width = 640
height = 480
fps = 30

video_writer = cv2.VideoWriter(
    "/home/dexter/Documents/GitHub/pose-to-biped/assets/robot_walk.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Optional camera setup
cam = mujoco.MjvCamera()
cam.distance = 5.0
cam.azimuth = 90
cam.elevation = -20
# ------------------------
# Rollout
# ------------------------
with mujoco.Renderer(vec_env.sim.model, height, width) as renderer:
    print("RENDERER RUNNING")
    for frame_idx in range(200):  # record N frames
        print(frame_idx)
        action, _ = model.predict(obs)

        obs, rewards, dones, trunc, info = vec_env.step(action)

        if dones:
            obs, _ = vec_env.reset()

        landmarks = vec_env.landmarks
        vec_env.sim.set_points([
            landmarks[16],
            landmarks[15],
            landmarks[28],
            landmarks[27],
            landmarks[12],
            landmarks[11]
        ])

        # Update scene
        renderer.update_scene(vec_env.sim.data, camera=cam)

        # Render image
        frame = renderer.render()

        # MuJoCo gives RGB, OpenCV needs BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write frame
        video_writer.write(frame)

    # Cleanup
    video_writer.release()
    renderer.close()

    print("Saved to robot_walk.mp4")