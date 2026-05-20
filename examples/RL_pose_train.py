import sys 
import os 
sys.path.append("/home/dexter/Documents/GitHub/pose-to-biped/") #replace with your file path
import cv2
import numpy as np
from sim.environment import * 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

vec_env = robo_gym(
    path_to_scene="/home/dexter/Documents/GitHub/pose-to-biped/Robots/scene.xml",
    path_to_urdf="/home/dexter/Documents/GitHub/pose-to-biped/Robots/h1_with_hand.urdf"
)
vec_env.set_dataset("/home/dexter/Documents/GitHub/pose-to-biped/models/")
vec_env._set_log("/home/dexter/Documents/GitHub/pose-to-biped/models/results/log")
model = PPO(
    "MlpPolicy",
    vec_env,
    device="cpu",
    learning_rate=5e-5,          # Lower learning rate prevents aggressive updates
    max_grad_norm=0.5,           # Forces severe gradients to be clipped
    clip_range=0.1,              # Limits policy changes per step
    clip_range_vf=0.1,           # CRITICAL: Prevents value function gradient explosions
    ent_coef=0.001,              # Decreases chaotic exploration noise
    
    # Custom policy architecture to stabilize continuous robot actions
    policy_kwargs=dict(
        log_std_init=-2.0,       # Start with smaller, tighter random movements
        ortho_init=True          # Scales down weights at initialization
    ),
    verbose=1,
    n_steps=256
)
checkpoint_callback = CheckpointCallback(
    save_freq=10000,          # save every 10k steps
    save_path="./logs/",
    name_prefix="/home/dexter/Documents/GitHub/pose-to-biped/models/test2"
)
model.learn(total_timesteps=250000)
model.save("/home/dexter/Documents/GitHub/pose-to-biped/models/test2")

del model # remove to demonstrate saving and loading

model = PPO.load("/home/dexter/Documents/GitHub/pose-to-biped/models/test2")

"""
import mujoco
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
        if dones: vec_env.reset()
        vec_env.sim.set_points([landmarks[16],landmarks[15],landmarks[28],landmarks[27],landmarks[12],landmarks[11]])
        viewer.sync()"""

