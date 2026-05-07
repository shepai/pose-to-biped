# Code to bring the mujuco environment to match the gym environment properties 
# Build mujuco environmet 
# step function must load movement pose frmo dataset and reset everytime the pose changes 
# 
from sim import *
from sim.kinematics import kinematics_tranfser 
import gymnasium as gym
from pose import PoseExtractor, PARENTS

class dataset:
    def __init__(self):
        self.X=None 
        self.ind=None
        self.converted=False 
        self.i=0
        self.ind_count=0
    def open_converted(self,filepath):
        self.X=np.load(filepath+"/X.npy")
        self.ind=np.load(filepath+"/IND.npy")
        self.converted=True 
    def next_landmarks(self):
        changed=False
        if self.i==self.ind[self.ind_count]:
            self.ind_count+=1
            print("INDEX CHANGE")
            changed=True
        if self.i>len(self.X):
            self.i=0 
        pose=self.X[self.i]
        self.i+=1
        return pose,changed
    def current_landmarks(self):
        return self.X[self.i]
class robo_gym(gym.Env):
    def __init__(self,path_to_scene="C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml",
                 path_to_urdf="/its/home/drs25/unitree_ros/robots/h1_description/urdf/h1_with_hand.urdf"): #cannot remember how many joints
        self.sim = MujocoSimulator(path_to_scene)
        self.ki_mod=kinematics_tranfser(path_to_urdf)
        self.pose=PoseExtractor()
        num_joints = int(np.asarray(self.sim.get_position()).shape[0])
        self.imu=self.sim.get_imu() 
        self.action_space = gym.spaces.Box(
            low=-0.3,  # limit correction magnitude (radians)
            high=0.3,
            shape=(num_joints,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_joints * 2 + num_joints + len(self.imu),),  # pos, vel, ref, IMU
            dtype=np.float32
        )
        self.current_coords=None
        self.current_positions=None 
        
        self.dataset=None 
        self.history=[]
        self.filename=".log"
        self.iter=0
    def _set_log(self,filename):
        self.filename=filename
    def set_dataset(self,filepath):
        self.dataset=dataset()
        self.dataset.open_converted(filepath)
    def save(self):
        np.save(self.filename,np.array(self.history))
    def step(self,correction):
        landmarks,ind=self.dataset.next_landmarks()
        landmarks,_=self.pose.to_local_space(landmarks)
        landmarks=landmarks[:,:3] 
        landmarks=self.sim.align_human_to_robot(landmarks)
        if ind:
            self.sim.rotate_robot_to_human(landmarks)
        self.current_coords=[landmarks[14],landmarks[13],landmarks[28],landmarks[27]]
        trajectories=self.sim.get_trajectories(["right_wrist", "left_wrist", "right_ankle","left_ankle"],
                                          [landmarks[14],landmarks[13],landmarks[28],landmarks[27]])
        
        self.theta_ref=self.ki_mod.move_to(
                                    ["right_hand_link", "left_hand_link", "right_ankle_link","left_ankle_link"],
                                    targets=np.array(trajectories),
                                    max_iter=20
                                )
        #step through sim
        for dic in self.theta_ref:
            self.sim.map_move(dic,correction)
            # Update MuJoCo kinematics
            self.sim.set_step(5)     
        map=self.sim.get_coordinates()
        self.current_positions=[map["right_wrist"], map["left_wrist"], map["right_ankle"], map["left_ankle"]]
        reward = self._compute_reward()
        if self.iter%100==0: self.save()
        self.iter+=1
        terminated = self._check_fallen()
        return self._get_obs(), reward, terminated, False, {}
    def _compute_reward(self): 
        #get how close the joint positions are to the coordinates
        landmarks=self.dataset.current_landmarks()
        landmarks,_=self.pose.to_local_space(landmarks)
        landmarks=landmarks[:,:3]
        landmarks=self.sim.align_human_to_robot(landmarks)[[11, 12, 23, 24, 28, 27]]
        robot=np.array(list(self.sim.get_coordinates().values()))[[13, 18, 6, 1, 5, 10]]
        distances = np.linalg.norm(landmarks - robot, axis=1)
        avg_dist = np.mean(distances)
        tracking = -avg_dist
        fallen = self._check_fallen()
        stability = -10.0 if fallen else 0.5
        self.history.append(tracking+stability)
        return tracking + stability
    def _get_obs(self): 
        return np.concatenate([
            np.asarray(self.sim.get_position()),        # joint angles
            self.sim.get_velocities(),  # velocities (placeholder)
            np.zeros_like(self.current_joints),             # reference pose
            self.sim.get_imu()                 # IMU placeholder
        ]).astype(np.float32)
    def reset(self, seed=None, options=None):
        self.sim.reset()
        self.dataset.i = 0
        self.dataset.ind_count = 0
        super().reset(seed=seed)
        self.current_joints = np.zeros_like(self.action_space.sample())
        self.theta_ref = np.zeros_like(self.current_joints)
        return self._get_obs(), {}
    def _check_fallen(self):
        # if too far away from points to really recover
        landmarks=self.dataset.current_landmarks()
        landmarks,_=self.pose.to_local_space(landmarks)
        landmarks=landmarks[:,:3]
        landmarks=self.sim.align_human_to_robot(landmarks)[[11, 12, 23, 24, 28, 27]]
        robot=np.array(list(self.sim.get_coordinates().values()))[[13, 18, 6, 1, 5, 10]]
        distances = np.linalg.norm(landmarks - robot, axis=1)
        avg_dist = np.mean(distances)
        return avg_dist > 0.4