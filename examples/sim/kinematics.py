import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
import pink
from pink import solve_ik
from pink.tasks import ComTask, FrameTask
from difflib import get_close_matches  # standard library fuzzy matching

class kinematics_tranfser:
    def __init__(self,path_ro_urdf): #set up humanoid chassis
        self.model = pin.buildModelFromUrdf(path_ro_urdf)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)  # default joint positions
        self.configuration = pink.Configuration(self.model, self.data, self.q)
        self.com_task = ComTask(cost=200.0)
        self.com_task.set_target_from_configuration(self.configuration)
        self.joint_map = {
    "left_hip_yaw": "left_hip_yaw_joint",
    "left_hip_roll": "left_hip_roll_joint",
    "left_hip_pitch": "left_hip_pitch_joint",
    "left_knee": "left_knee_joint",
    "left_ankle": "left_ankle_joint",

    "right_hip_yaw": "right_hip_yaw_joint",
    "right_hip_roll": "right_hip_roll_joint",
    "right_hip_pitch": "right_hip_pitch_joint",
    "right_knee": "right_knee_joint",
    "right_ankle": "right_ankle_joint",

    "torso": "torso_joint",

    "left_shoulder_pitch": "left_shoulder_pitch_joint",
    "left_shoulder_roll": "left_shoulder_roll_joint",
    "left_shoulder_yaw": "left_shoulder_yaw_joint",
    "left_elbow": "left_elbow_joint",

    "right_shoulder_pitch": "right_shoulder_pitch_joint",
    "right_shoulder_roll": "right_shoulder_roll_joint",
    "right_shoulder_yaw": "right_shoulder_yaw_joint",
    "right_elbow": "right_elbow_joint",
}
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
                damping=0.05,
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
    def equalise_sims(self, mujoco_sim):
        mujoco_joints = mujoco_sim.get_coordinates()

        # IMPORTANT: start from current Pinocchio state
        q = np.array(self.configuration.q, copy=True)

        for mujoco_name, pin_name in self.joint_map.items():

            if mujoco_name not in mujoco_joints:
                continue

            if pin_name not in self.model.names:
                continue

            mujoco_q = np.atleast_1d(mujoco_joints[mujoco_name])

            joint_id = self.model.getJointId(pin_name)
            joint = self.model.joints[joint_id]

            q_start = joint.idx_q
            q_size = joint.nq

            # Safety check: dimension must match
            if len(mujoco_q) != q_size:
                continue

            q[q_start:q_start + q_size] = mujoco_q

        # update configuration
        self.configuration.q = q

        pin.forwardKinematics(self.model, self.data, q)
        self.configuration.update()
if __name__=="__main__":
    import os
    os.environ["MUJOCO_GL"] = "egl"
    import mujoco
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
        #ki_mod.equalise_sims(sim)
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
                ki_mod.equalise_sims(sim)
            frame_id += 1
        