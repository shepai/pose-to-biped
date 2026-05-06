import sys 
import os 
sys.path.append("/home/dexter/Documents/GitHub/pose-to-biped/") #replace with your file path
import cv2
import numpy as np
from sim.environment import * 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = robo_gym(
    path_to_scene="/home/dexter/Documents/GitHub/pose-to-biped/Robots/scene.xml",
    path_to_urdf="/home/dexter/Documents/GitHub/pose-to-biped/Robots/h1_with_hand.urdf"
)
vec_env.set_dataset("/home/dexter/Documents/GitHub/pose-to-biped/models/")
model = PPO.load("/home/dexter/Documents/GitHub/pose-to-biped/models/test1", device="cpu")

obs,_ = vec_env.reset()
with mujoco.viewer.launch_passive(vec_env.sim.model, vec_env.sim.data) as viewer: 
    viewer.cam.distance = 5.0       
    while True:
        action, _states = model.predict(obs)
        landmarks=vec_env.dataset.current_landmarks()
        landmarks,_=vec_env.pose.to_local_space(landmarks)
        landmarks=landmarks[:,:3] 
        landmarks=vec_env.sim.align_human_to_robot(landmarks)
        obs, rewards, dones, trunc,info = vec_env.step(action)
        vec_env.sim.set_points([landmarks[16],landmarks[15],landmarks[28],landmarks[27],landmarks[12],landmarks[11]])
        viewer.sync()


