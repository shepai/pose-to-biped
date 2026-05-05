import mujoco
import mujoco.viewer
import numpy as np
np.set_printoptions(suppress=True)
import logging
from scipy.spatial.transform import Rotation as Rot
logging.getLogger().setLevel(logging.ERROR)

class MujocoSimulator:
    def __init__(self, xml_path: str,gravity=True):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        print("Number of joints:", self.model.njnt)
        print("Number of degrees of freedom (qpos):", self.model.nq)
        print("Number of actuators:", self.model.nu)
        joint_start = self.model.nq - self.model.nu
        joint_end = self.model.nq
        self.initial=self.data.qpos[:]
        self.initial = self.initial[joint_start:joint_end].copy()
        self.mapping={}
        self.names=[]
        self.gravity=gravity
        self.transform = False
        self.s = None
        self.R = None
        self.t = None
        if not gravity:
            #self.model.opt.gravity[:] = [0, 0, 0]
            pass
        for i in range(self.model.njnt):
            self.names=self.model.joint(i).name
        self.transform=False
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        joint_start = self.model.nq - self.model.nu
        joint_end = self.model.nq
        self.data.qpos[:]=self.initial[joint_start:joint_end].copy()
        if not self.gravity:
            self.model.opt.gravity[:] = [0, 0, 0]
    def set_position(self, target_qpos, kp=200.0, kd=50.0):
        joint_qpos_start = self.model.nq - self.model.nu
        joint_qvel_start = self.model.nv - self.model.nu

        pos_error = target_qpos - self.data.qpos[joint_qpos_start:]
        vel_error = -self.data.qvel[joint_qvel_start:]

        torque = kp * pos_error + kd * vel_error
        self.data.ctrl[:] = torque
        
    def get_position(self):
        joint_qpos_start = self.model.nq - self.model.nu
        return self.data.qpos[joint_qpos_start:]
    def set_step(self, n_steps: int = 1):
        """
        Advance the simulation by n steps.
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
    def get_coordinates(self):
        self.mapping={}
        for j in range(self.model.njnt):
            joint = self.model.joint(j)
            position = self.data.xanchor[j]  # world position of the body
            self.mapping[joint.name] = position
        return self.mapping
    def get_local_coordinates(self):
        """
        Get the centre point, and map all limb coordinates to local
        """
        #get joint names 
        self.mapping=self.get_coordinates()
        #define centre point
        p1=self.mapping['right_hip_roll']
        p2=self.mapping['left_hip_roll']
        centre= (p1 + p2) / 2.0
        #recalculate other points
        for key in self.mapping:
            self.mapping[key]=self.mapping[key]-centre
        return self.mapping #return points and centre
    def gethips(self):
        self.mapping=self.get_coordinates()
        p1=self.mapping['right_hip_roll']
        p2=self.mapping['left_hip_roll']
        return (p1 + p2) / 2.0
    def get_trajectories(self,names,coords,alpha=0.4): #get the trajectory between specific points
        traj=[]
        self.mapping=self.get_coordinates()
        for i in range(len(names)):
            v=coords[i]-self.mapping[names[i]]
            traj.append(v)
        for i in range(len(traj)):
            traj[i] = alpha * traj[i] + (1 - alpha) * traj[i]
        return traj
    def get_coords_of(self,names):
        self.mapping=self.get_coordinates()
        return np.array([self.mapping[names[i]] for i in range(len(names))])
    def convert_normal_coordinates(self,coords):
        #get joint names 
        for j in range(self.model.njnt):
            joint = self.model.joint(j)
            body_id = self.model.jnt_bodyid[j]
            position = self.data.xpos[body_id]  # world position of the body
            self.mapping[joint.name] = position
        #define centre point
        p1=self.mapping['right_hip_roll']
        p2=self.mapping['left_hip_roll']
        centre= (p1 + p2) / 2.0
        for key in range(len(coords)):
            coords[key]=coords[key]+centre
        return coords #return points and centre
    def get_state(self):
        state = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "act": self.data.act.copy() if self.data.act is not None else None
        }
        return state
    def map_move(self, joint_dict):
        for name, value in joint_dict.items():
            mj_name = name.replace("_joint", "")

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                mj_name
            )

            if joint_id != -1:
                qpos_adr = self.model.jnt_qposadr[joint_id]

                self.data.qpos[qpos_adr:qpos_adr + len(np.atleast_1d(value))] = value
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
    def set_state(self,state):
        self.data.qpos=state["qpos"]
        self.data.qvel=state["qvel"]
        self.data.act=state["act"]
    def align_human_to_robot(self, human_pose):
        robot_coords = np.array(list(self.get_coordinates().values()))
        H = human_pose[[11, 12, 23, 24, 28, 27]].astype(np.float64)
        R = robot_coords[[13, 18, 6, 1, 5, 10]].astype(np.float64)
        # center both
        H_mean = H.mean(axis=0)
        R_mean = R.mean(axis=0)
        Hc = H - H_mean
        Rc = R - R_mean
        # SCALE ONLY (no rotation)
        scale = np.sqrt(np.sum(Rc ** 2) / np.sum(Hc ** 2))
        # apply scaling only
        H_scaled = human_pose.astype(np.float64) * scale
        # translate to robot centroid
        aligned = H_scaled + (R_mean - H_mean * scale)
        return aligned
    def rotate_robot_to_human(self, human_pose):
        robot_coords = np.array(list(self.get_coordinates().values()), dtype=np.float64)

        # Corresponding joint subsets
        H = human_pose[[11, 12, 23, 24, 28, 27]].astype(np.float64)
        R = robot_coords[[13, 18, 6, 1, 5, 10]].astype(np.float64)

        # --- 1. Center both point sets ---
        H_mean = H.mean(axis=0)
        R_mean = R.mean(axis=0)

        Hc = H - H_mean
        Rc = R - R_mean

        # --- 2. Kabsch alignment (optimal rotation) ---
        Hcov = Hc.T @ Rc
        U, S, Vt = np.linalg.svd(Hcov)

        R_opt = Vt.T @ U.T

        # Fix reflection issue (ensure proper rotation)
        if np.linalg.det(R_opt) < 0:
            Vt[-1, :] *= -1
            R_opt = Vt.T @ U.T

        # --- 3. Convert rotation matrix to Euler angles ---
        rot_obj = Rot.from_matrix(R_opt)
        euler_xyz = rot_obj.as_euler('xyz', degrees=False)

        # Debug (optional)
        print("Euler (deg):", np.degrees(euler_xyz))

        # --- 4. Apply to robot ---
        self.set_orientation(euler_xyz)

    def set_orientation(self, angle):
        angles_rad = angle#np.deg2rad(angle) # angles = [roll, pitch, yaw]
        self.data.eq_active[:] = 0
        mujoco.mj_resetData(self.model, self.data) 
        target_quat = np.zeros(4)
        mujoco.mju_euler2Quat(target_quat, angles_rad, 'xyz')
        
        # 4. Apply to the robot base
        self.data.qpos[3:7] = target_quat
        
        # 5. Stop physics glitches
        self.data.qvel[:] = 0 
        mujoco.mj_forward(self.model, self.data)
        if self.model.nmocap > 0:
            self.data.mocap_quat[0] = target_quat
            # Also update position if you moved it
            self.data.mocap_pos[0] = self.data.qpos[0:3] 
        self.align_to_floor()
        self.data.eq_active[:] = 1
        joint_start = self.model.nq - self.model.nu
        joint_end = self.model.nq
        self.initial=self.data.qpos[:]
        self.initial = self.initial[joint_start:joint_end].copy()
    def get_robot_lowest_point(self):
        mujoco.mj_forward(self.model, self.data)
        min_z = float('inf')
        for i in range(self.model.ngeom):
            # skip world geoms if needed
            body_id = self.model.geom_bodyid[i]
            if body_id == 0:
                continue
            pos = self.data.geom_xpos[i]
            z = pos[2]
            gtype = self.model.geom_type[i]
            size = self.model.geom_size[i]
            # approximate vertical extent in LOCAL frame
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                bottom = z - size[0]
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                bottom = z - size[1]
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                bottom = z - size[1]
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                bottom = z - size[2]
            else:
                bottom = z
            min_z = min(min_z, bottom)
        return min_z
    def align_to_floor(self, offset=0.005):
        mujoco.mj_forward(self.model, self.data)
        min_z = self.get_robot_lowest_point()
        shift = -min_z + offset
        self.data.qpos[2] += shift
        mujoco.mj_forward(self.model, self.data)
        
    def set_pose(self,angles):
        pass


# Example usage
if __name__ == "__main__":
    sim = MujocoSimulator(
        "C:/Users/dexte/Documents/GitHub/pose-to-biped/Robots/scene.xml"
    )
    j=0
    
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.distance = 5.0
            sim.set_orientation([00,00,00])
            while viewer.is_running():
                j+=1
                #if j<400:
                    #sim.set_position(sim.initial+(j/10000))
                sim.set_step()
                viewer.sync()
                #print(sim.get_local_coordinates())