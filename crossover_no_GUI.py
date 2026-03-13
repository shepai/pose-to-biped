"""
This versionn of the code is designed to be run over SSH to test the functions. 
"""
from pose import PoseExtractor, PARENTS
from sim import MujocoSimulator
from sim.kinematics import kinematics_tranfser 
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import cv2
if __name__ == "__main__":
    extractor = PoseExtractor(missing_value=-1.0)
    cap = cv2.VideoCapture(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sim = MujocoSimulator(
            "C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml"
        )
    j=0
    ki_mod=kinematics_tranfser("/its/home/drs25/unitree_ros/robots/h1_description/urdf/h1_with_hand.urdf")
    max_w = sim.model.vis.global_.offwidth
    max_h = sim.model.vis.global_.offheight
    renderer = mujoco.Renderer(sim.model, height=max_h, width=max_w)
    while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = extractor.process(frame)
            plt.cla()
            ax = extractor.plot_world_landmarks(landmarks, ax)
            #get the hand and ankle links
            movements=ki_mod.move_to(["right_hand_link", "left_hand_link", "right_ankle_link","left_ankle_link"],
            targets=np.array([landmarks[16],landmarks[15],landmarks[28],landmarks[27]]))
            #step through sim
            for dic in movements:
                sim.map_move(dic)
                # Update MuJoCo kinematics
                for i in range(1):
                    sim.set_step(10)     
            plt.pause(0.005)
            renderer.update_scene(sim.data)
            pixels = renderer.render()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            frame = np.concatenate((frame, pixels), axis=1)
            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            j+=1
             
    cap.release()
    extractor.close()
    cv2.destroyAllWindows()