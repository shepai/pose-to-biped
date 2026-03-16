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
import matplotlib.pyplot as plt

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
    out = cv2.VideoWriter(output_path, fourcc, 25, (640*3, 480))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    while j<100:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = extractor.process(frame)
        landmarks,_=extractor.to_local_space(landmarks)
        hips=sim.gethips()
        landmarks=landmarks[:,:3] 
        landmarks=(landmarks+hips) 
        landmarks=sim.align_human_to_robot(landmarks,np.array(list(sim.get_coordinates().values())))
        ax.cla()
        ax=extractor.plot_world_landmarks(landmarks,ax,
                                          points=np.array(list(sim.get_coordinates().values())))#sim.get_coords_of(["right_elbow", "left_elbow", "right_ankle","left_ankle"]))
        #get the hand and ankle links
        trajectories=sim.get_trajectories(["right_wrist", "left_wrist", "right_ankle","left_ankle"],
                                          [landmarks[14],landmarks[13],landmarks[28],landmarks[27]])
        #trajectories=[landmarks[14],landmarks[13],landmarks[28],landmarks[27]]
        movements = ki_mod.move_to(
                                    ["right_hand_link", "left_hand_link", "right_ankle_link","left_ankle_link"],
                                    targets=np.array(trajectories),
                                    max_iter=100
                                )
        #step through sim
        for dic in movements:
            sim.map_move(dic)
            # Update MuJoCo kinematics
            sim.set_step(5)     
        renderer.update_scene(sim.data)
        pixels = renderer.render()
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (640, 480))  # match webcam frame
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.resize(img, (640, 480))
        frame = np.concatenate((frame, img, pixels), axis=1).astype(np.uint8)
        out.write(frame)
        cv2.imwrite("debug_frame.png", frame)
        j += 1
        print(f"Processed frame {j}")
        #ki_mod.equalise_sims(sim)
    renderer.close()
    out.release()       
    cap.release()
    extractor.close()