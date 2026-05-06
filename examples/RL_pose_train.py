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
vec_env._set_log("/home/dexter/Documents/GitHub/pose-to-biped/models/results/log")
model = PPO("MlpPolicy", vec_env, device="cpu",verbose=1,n_steps=256)
model.learn(total_timesteps=250000)
model.save("/home/dexter/Documents/GitHub/pose-to-biped/models/test1")

del model # remove to demonstrate saving and loading

model = PPO.load("/home/dexter/Documents/GitHub/pose-to-biped/models/test1")

obs = vec_env.reset()
while True:
    
    action, _states = model.predict(obs)
    obs, rewards, dones, trunc,info = vec_env.step(action)
    vec_env.render("human")


