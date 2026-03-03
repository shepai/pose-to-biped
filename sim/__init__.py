import mujoco
import mujoco.viewer


class MujocoSimulator:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        print("Number of joints:", self.model.njnt)
        print("Number of degrees of freedom (qpos):", self.model.nq)
        print("Number of actuators:", self.model.nu)
    def set_position(self, qpos):
        """
        Set the generalized positions.
        qpos should match the model's joint dimension.
        """
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def set_step(self, n_steps: int = 1):
        """
        Advance the simulation by n steps.
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def run(self):
        """
        Launch the passive viewer and run the simulation loop.
        """
        self.set_position([0 for i in range(26)])
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


# Example usage
if __name__ == "__main__":
    sim = MujocoSimulator(
        "C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml"
    )
    
    sim.run()