import mujoco
import mujoco.viewer
import numpy as np

class MujocoSimulator:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        print("Number of joints:", self.model.njnt)
        print("Number of degrees of freedom (qpos):", self.model.nq)
        print("Number of actuators:", self.model.nu)
        joint_start = self.model.nq - self.model.nu
        joint_end = self.model.nq
        self.initial=self.data.qpos[:]
        self.initial = self.initial[joint_start:joint_end]
        self.mappting={}
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
        pass 
        #get joint names 
        #define centre point
        #recalculate other points
        return 0 #return points and centre
        
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


# Example usage
if __name__ == "__main__":
    sim = MujocoSimulator(
        "C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml"
    )
    j=0
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                j+=1
                #sim.set_position(sim.initial+(j/10000))
                mujoco.mj_step(sim.model, sim.data)
                viewer.sync()