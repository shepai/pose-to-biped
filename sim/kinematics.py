import os
os.environ["MUJOCO_GL"] = "egl"
import mujoco
import mujoco.viewer
from __init__ import MujocoSimulator
import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
import pink
from pink import solve_ik
from pink.tasks import ComTask, FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
import imageio

sim = MujocoSimulator(
    "/its/home/drs25/mujoco-menagerie-main/unitree_h1/scene.xml",
    gravity=0
)
model = pin.buildModelFromUrdf("/its/home/drs25/unitree_ros/robots/h1_description/urdf/h1_with_hand.urdf")
data = model.createData()
q = pin.neutral(model)  # default joint positions

configuration = pink.Configuration(model, data, q)
pelvis_orientation_task = FrameTask(
        "pelvis",
        position_cost=0.0,  # [cost] / [m]
        orientation_cost=10.0,  # [cost] / [rad]
    )

com_task = ComTask(cost=200.0)
com_task.set_target_from_configuration(configuration)

posture_task = PostureTask(
    cost=1e-1,  # [cost] / [rad]
)

tasks = [pelvis_orientation_task, posture_task, com_task]

for arm_points in ["right_hand_link", "left_hand_link"]:
    task = FrameTask(
        arm_points,
        position_cost=4.0,  # [cost] / [m]
        orientation_cost=0.0,  # [cost] / [rad]
    )
    tasks.append(task)

for task in tasks:
    task.set_target_from_configuration(configuration)
    if isinstance(task, FrameTask):
        target = task.transform_target_to_world
        if task.frame in ["right_hand_link", "left_hand_link"]:
            target.translation += np.array([-0.1, 0.0, -0.2])
            task.set_target(target)

# Select QP solver
solver = qpsolvers.available_solvers[0]
if "osqp" in qpsolvers.available_solvers:
    solver = "osqp"

rate = RateLimiter(frequency=200.0, warn=False)
dt = rate.period
t = 0.0  # [s]
period = 2
omega = 2 * np.pi / period
max_w = sim.model.vis.global_.offwidth
max_h = sim.model.vis.global_.offheight
renderer = mujoco.Renderer(sim.model, height=max_h, width=max_w)
frame_id = 0
save_every = 1000   # save every 50 simulation steps
while True:
        # Update CoM target
        Az = 0.05
        desired_com = np.zeros(3)
        desired_com[2] = 0.55 + Az * np.sin(omega * t)
        com_task.set_target(desired_com)

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            damping=0.01,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)

        rate.sleep()
        t += dt
        dic = {}
        for joint in model.names:  # skip universe
            joint_id = model.getJointId(joint)
            q_start = model.joints[joint_id].idx_q
            q_size = model.joints[joint_id].nq
            joint_q = configuration.q[q_start : q_start + q_size]
            dic[joint]=joint_q
        sim.map_move(dic)

        # Update MuJoCo kinematics
        for i in range(10):
            mujoco.mj_forward(sim.model, sim.data)
        
        renderer.update_scene(sim.data)
        pixels = renderer.render()

        if frame_id % save_every == 0 and frame_id<10000:
            imageio.imwrite(f"/its/home/drs25/pose-to-biped/assets/snapshots/frame_{frame_id:05d}.png", pixels)

        frame_id += 1