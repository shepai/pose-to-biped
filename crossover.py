"""
This versionn of the code is designed to be run on device
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
            "/home/dexter/Documents/GitHub/pose-to-biped/Robots/scene.xml"
        )
    ki_mod=kinematics_tranfser("/home/dexter/Documents/GitHub/pose-to-biped/Robots/h1_with_hand.urdf")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:        
        while viewer.is_running():
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = extractor.process(frame)
            landmarks,_=extractor.to_local_space(landmarks)
            hips=sim.gethips()
            if landmarks is not None:
                landmarks=landmarks[:,:3] 
                landmarks=(landmarks+hips) 
                landmarks=sim.align_human_to_robot(landmarks,np.array(list(sim.get_coordinates().values())))
                #ax.cla()
                #ax=extractor.plot_world_landmarks(landmarks,ax,points=np.array(list(sim.get_coordinates().values())))#sim.get_coords_of(["right_elbow", "left_elbow", "right_ankle","left_ankle"]))
                #get the hand and ankle links
                trajectories=sim.get_trajectories(["right_wrist", "left_wrist", "right_ankle","left_ankle"],
                                                [landmarks[16],landmarks[15],landmarks[28],landmarks[27]])
                #trajectories=[landmarks[14],landmarks[13],landmarks[28],landmarks[27]]
                movements = ki_mod.move_to(
                                            ["right_hand_link", "left_hand_link", "right_ankle_link","left_ankle_link"],
                                            targets=np.array(trajectories),
                                            max_iter=20
                                        )
                for dic in movements:
                    sim.map_move(dic)
                    # Update MuJoCo kinematics
                    sim.set_step(1)     
            viewer.sync()
            #frame = cv2.resize(frame, (640, 480))  # match webcam frame
            #fig.canvas.draw()
            #img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            #img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

            #img = img[..., :3]  # convert RGBA → RGB
            #img = cv2.resize(img, (640, 480))
            #frame = np.concatenate((frame), axis=1).astype(np.uint8)
            cv2.imshow("debug_frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
             
    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
