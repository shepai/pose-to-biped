import sys 
import os 
sys.path.append("/home/dexter/Documents/GitHub/pose-to-biped/") #replace with your file path
import cv2
import numpy as np
from sim.environment import * 
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_util import make_vec_env

vec_env = robo_gym(
    path_to_scene="/home/dexter/Documents/GitHub/pose-to-biped/Robots/scene.xml",
    path_to_urdf="/home/dexter/Documents/GitHub/pose-to-biped/Robots/h1_with_hand.urdf"
)
vec_env.set_dataset("/home/dexter/Documents/GitHub/pose-to-biped/models/")
model = PPO.load("/home/dexter/Documents/GitHub/pose-to-biped/models/ppo/test_ppo_2100000_steps.zip", device="cpu")
#model = SAC.load("/home/dexter/Documents/GitHub/pose-to-biped/models/sac/test_SAC_5000000_steps.zip", device="cpu")


obs,_ = vec_env.reset()
with mujoco.viewer.launch_passive(vec_env.sim.model, vec_env.sim.data) as viewer: 
    viewer.cam.distance = 5.0       
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, trunc,info = vec_env.step(action) #np.zeros_like(action)
        if dones: vec_env.reset()
        landmarks=vec_env.landmarks
        vec_env.sim.set_points([landmarks[16],landmarks[15],landmarks[28],landmarks[27],landmarks[12],landmarks[11]])
        viewer.sync()
        


