# Code to bring the mujuco environment to match the gym environment properties 
# Build mujuco environmet 
# step function must load movement pose frmo dataset and reset everytime the pose changes 
# 
from __init__ import *
from kinematics import kinematics_tranfser 
import gymnasium as gym

class robo_gym(gym.Env):
    def __init__(self,path_to_scene="C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml",
                 path_to_urdf="/its/home/drs25/unitree_ros/robots/h1_description/urdf/h1_with_hand.urdf",
                 num_joints=29): #cannot remember how many joints
        self.sim = MujocoSimulator(path_to_scene)
        self.ki_mod=kinematics_tranfser(path_to_urdf)
        self.action_space = gym.spaces.Box(
            low=-0.3,  # limit correction magnitude (radians)
            high=0.3,
            shape=(num_joints,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_joints * 2 + num_joints + 6,),  # pos, vel, ref, IMU
            dtype=np.float32
        )
        self.current_coords=None
        self.current_positions=None 
        self.imu=None
    def step(self,correction):
        landmarks=next_landmarks()
        self.current_coords=[landmarks[14],landmarks[13],landmarks[28],landmarks[27]]
        trajectories=sim.get_trajectories(["right_wrist", "left_wrist", "right_ankle","left_ankle"],
                                          [landmarks[14],landmarks[13],landmarks[28],landmarks[27]])
        
        self.theta_ref=self.ki_mod.move_to(
                                    ["right_hand_link", "left_hand_link", "right_ankle_link","left_ankle_link"],
                                    targets=np.array(trajectories),
                                    max_iter=100
                                )
        theta_final = self.theta_ref + correction
        #step through sim
        for dic in theta_final:
            sim.map_move(dic)
            # Update MuJoCo kinematics
            sim.set_step(5)     
        map=self.sim.get_coordinates()
        self.current_positions=[map["right_wrist"], map["left_wrist"], map["right_ankle"], map["left_ankle"]]
        reward = self._compute_reward()
        terminated = self._check_fallen()
        return self._get_obs(), reward, terminated, False, {}
    def _compute_reward(self):
        #get how close the joint positions are to the coordinates
        #is the robot stable?
        balance_reward = ...
        tracking_reward = -0.1 * np.linalg.norm(self.theta_ref - self.current_joints)
        return balance_reward + tracking_reward