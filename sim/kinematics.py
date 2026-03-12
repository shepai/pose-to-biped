import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
import pink
from pink import solve_ik
from pink.tasks import ComTask, FrameTask

class kinematics_tranfser:
    def __init__(self,path_ro_urdf): #set up humanoid chassis
        self.model = pin.buildModelFromUrdf(path_ro_urdf)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)  # default joint positions
        self.configuration = pink.Configuration(self.model, self.data, self.q)
        self.com_task = ComTask(cost=200.0)
        self.com_task.set_target_from_configuration(self.configuration)

    def move_to(self,joint_names=["right_hand_link", "left_hand_link"],targets=np.array([[-0.1, 0.1, 0.5],[-0.1, 0.1, 0.5]]),max_iter=100): #calculate joint movements
        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.period
        t = 0.0  # [s]
        period = 2
        omega = 2 * np.pi / period
        tasks = [] #pelvis_orientation_task, 
        for arm_points in joint_names:
            task = FrameTask(
                arm_points,
                position_cost=4.0,  # [cost] / [m]
                orientation_cost=0.0,  # [cost] / [rad]
            )
            tasks.append(task)
        for j,task in enumerate(tasks):
            task.set_target_from_configuration(self.configuration)
            if isinstance(task, FrameTask):
                target = task.transform_target_to_world
                if task.frame in joint_names:
                    target.translation += targets[j]
                    task.set_target(target)
        solver = qpsolvers.available_solvers[0]
        if "osqp" in qpsolvers.available_solvers:
            solver = "osqp"
        movements=[]
        for i in range(max_iter):
            Az = 0.05
            desired_com = np.zeros(3)
            desired_com[2] = 0.55 + Az * np.sin(omega * t)
            self.com_task.set_target(desired_com)

            velocity = solve_ik(
                self.configuration,
                tasks,
                dt,
                solver=solver,
                damping=0.01,
                safety_break=False,
            )
            self.configuration.integrate_inplace(velocity, dt)

            rate.sleep()
            t += dt
            dic = {}
            for joint in self.model.names:  # skip universe
                joint_id = self.model.getJointId(joint)
                q_start = self.model.joints[joint_id].idx_q
                q_size = self.model.joints[joint_id].nq
                joint_q = self.configuration.q[q_start : q_start + q_size]
                dic[joint]=joint_q
            movements.append(dic)
        return movements
    def get_transformations(self): #return the suggested movement array
        pass 
    def equalise_sims(self): #make sure the simulations both align at the end of each
        pass 
    

if __name__=="__main__":
    import os
    os.environ["MUJOCO_GL"] = "egl"
    import mujoco
    import mujoco.viewer
    import imageio
    from __init__ import MujocoSimulator
    sim = MujocoSimulator(
        "/its/home/drs25/mujoco-menagerie-main/unitree_h1/scene.xml",
        gravity=0
    )
    ki_mod=kinematics_tranfser("/its/home/drs25/unitree_ros/robots/h1_description/urdf/h1_with_hand.urdf")
    max_w = sim.model.vis.global_.offwidth
    max_h = sim.model.vis.global_.offheight
    renderer = mujoco.Renderer(sim.model, height=max_h, width=max_w)
    frame_id = 0
    save_every = 1000   # save every 50 simulation steps
    while True:
        movements=ki_mod.move_to(["right_hand_link", "left_hand_link", "right_ankle_link","left_ankle_link"],
            targets=np.array([[0, -0.01, 0],[0, 0.01, 0],[0, 0.01, 0.05],[0, 0.01, 0]]))
        # Update CoM target
        for dic in movements:
            sim.map_move(dic)
            # Update MuJoCo kinematics
            for i in range(1):
                mujoco.mj_forward(sim.model, sim.data)
            renderer.update_scene(sim.data)
            pixels = renderer.render()
            if frame_id % save_every == 0 and frame_id<10000:
                imageio.imwrite(f"/its/home/drs25/pose-to-biped/assets/snapshots/frame_{frame_id:05d}.png", pixels)
            frame_id += 1