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

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")


