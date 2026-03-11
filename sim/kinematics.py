import mujoco
import mujoco.viewer
import numpy as np
from __init__ import MujocoSimulator
import time
import pin_pink as pink

sim = MujocoSimulator(
    "C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml",
    gravity=0
)

# Load URDF
robot = pink.RobotModel.from_urdf("C:/Users/dexte/Documents/Robots/unitree_ros-master/robots/h1_description/urdf/h1.urdf")
# Set end‑effector targets
targets = {
    "left_hand_link": np.array([0.35, 0.30, 1.25]),
    "right_hand_link": np.array([0.35,-0.30, 1.25])
}

# Solve IK
ik_solution = robot.compute_ik(targets)
# Apply to MuJoCo:
sim.data.qpos[:] = ik_solution.qpos