"""
This versionn of the code is designed to be run over SSH to test the functions. 
"""
import os
os.environ["MUJOCO_GL"] = "egl"
from pose import PoseExtractor, PARENTS
from sim import MujocoSimulator
from sim.kinematics import kinematics_tranfser 
import numpy as np
import mujoco
import cv2

if __name__ == "__main__":
    extractor = PoseExtractor(missing_value=-1.0)
    cap = cv2.VideoCapture("/its/home/drs25/pose-to-biped/assets/walking.mp4")
    sim = MujocoSimulator(
            "/its/home/drs25/mujoco-menagerie-main/unitree_h1/scene.xml"
        )
    max_w = sim.model.vis.global_.offwidth
    max_h = sim.model.vis.global_.offheight
    renderer = mujoco.Renderer(sim.model, height=max_h, width=max_w)
    j=0
    ki_mod=kinematics_tranfser("/its/home/drs25/unitree_ros/robots/h1_description/urdf/h1_with_hand.urdf")
    # Output video writer
    output_path = "/its/home/drs25/pose-to-biped/assets/output_record.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 25, (1280, 480))

    while j<10:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = extractor.process(frame)
        landmarks,_=extractor.to_local_space(landmarks)
        landmarks=landmarks[:,:3]
        #get the hand and ankle links
        movements=ki_mod.move_to(["right_hand_link", "left_hand_link", "right_ankle_link","left_ankle_link"],
        targets=np.array([landmarks[16],landmarks[15],landmarks[28],landmarks[27]]))
        #step through sim
        for dic in movements:
            sim.map_move(dic)
            # Update MuJoCo kinematics
            for i in range(1):
                sim.set_step(1)     
        renderer.update_scene(sim.data)
        pixels = renderer.render()
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (640, 480))  # match webcam frame
        frame = np.concatenate((frame, pixels), axis=1).astype(np.uint8)
        out.write(frame)
        cv2.imwrite("debug_frame.png", frame)
        j += 1
        print(f"Processed frame {j}")
    renderer.close()
    out.release()       
    cap.release()
    extractor.close()