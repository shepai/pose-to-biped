import mujoco
import mujoco.viewer
import mujoco

model = mujoco.MjModel.from_xml_path("C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()