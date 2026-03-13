import mujoco
import mujoco.viewer
import numpy as np

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
        self.initial = self.initial[joint_start:joint_end]
        self.mapping={}
        self.names=[]
        self.gravity=gravity
        if not gravity:
            #self.model.opt.gravity[:] = [0, 0, 0]
            pass
        for i in range(self.model.njnt):
            self.names=self.model.joint(i).name
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
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
    def get_local_coordinates(self):
        """
        Get the centre point, and map all limb coordinates to local
        """
        #get joint names 
        for j in range(self.model.njnt):
            joint = self.model.joint(j)
            body_id = self.model.jnt_bodyid[j]
            position = self.data.xpos[body_id]  # world position of the body
            self.mapping[joint.name]= position
        #define centre point
        p1=self.mapping['right_hip_roll']
        p2=self.mapping['left_hip_roll']
        centre= (p1 + p2) / 2.0
        #recalculate other points
        for key in self.mapping:
            self.mapping[key]=self.mapping[key]-centre
        return self.mapping #return points and centre
    def gethips(self):
        for j in range(self.model.njnt):
            joint = self.model.joint(j)
            body_id = self.model.jnt_bodyid[j]
            position = self.data.xpos[body_id]  # world position of the body
            self.mapping[joint.name]= position
        p1=self.mapping['right_hip_roll']
        p2=self.mapping['left_hip_roll']
        return (p1 + p2) / 2.0
    def get_trajectories(self,names,coords): #get the trajectory between specific points
        traj=[]
        for j in range(self.model.njnt):
            joint = self.model.joint(j)
            body_id = self.model.jnt_bodyid[j]
            position = self.data.xpos[body_id]  # world position of the body
            self.mapping[joint.name]= position
        for i in range(len(names)):
            v=self.mapping[names[i]]-coords[i]
            traj.append(v)
        return traj
    def convert_normal_coordinates(self,coords):
        #get joint names 
        for j in range(self.model.njnt):
            joint = self.model.joint(j)
            body_id = self.model.jnt_bodyid[j]
            position = self.data.xpos[body_id]  # world position of the body
            self.mapping[joint.name]= position
        #define centre point
        p1=self.mapping['right_hip_roll']
        p2=self.mapping['left_hip_roll']
        centre= (p1 + p2) / 2.0
        for key in range(len(coords)):
            coords[key]=coords[key]+centre
        return coords #return points and centre
    def run(self):
        """
        Launch the passive viewer and run the simulation loop.
        """
        j=0
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                j+=1
                self.set_position(self.initial+(j/10000))
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
    def get_state(self):
        state = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "act": self.data.act.copy() if self.data.act is not None else None
        }
        return state
    def map_move(self, joint_dict):
        for name, value in joint_dict.items():
            # Clean the name if your URDF names have "_joint" suffix but MuJoCo doesn't
            mj_name = name.replace("_joint", "")
            
            try:
                # Get the correct ID for this specific joint name
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
                
                if joint_id != -1:
                    # Get the starting address of this joint's data in qpos
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    # Map the value (ensure it's the right size, e.g., 1 for hinge, 7 for freejoint)
                    self.data.qpos[qpos_adr : qpos_adr + len(np.atleast_1d(value))] = value
            except ValueError:
                # This skips names that exist in Pinocchio but not in MuJoCo (like the frozen_part)
                continue
    def set_state(self,state):
        self.data.qpos=state["qpos"]
        self.data.qvel=state["qvel"]
        self.data.act=state["act"]


# Example usage
if __name__ == "__main__":
    sim = MujocoSimulator(
        "C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml"
    )
    j=0
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                j+=1
                if j<400:
                    sim.set_position(sim.initial+(j/10000))
                sim.set_step()
                viewer.sync()
                #print(sim.get_local_coordinates())