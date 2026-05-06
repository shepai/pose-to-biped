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

def get_traj(extractor,sim,frame):
    #global ax
    landmarks = extractor.process(frame)
    landmarks,_=extractor.to_local_space(landmarks)
    #hips=sim.gethips()
    if landmarks is not None:
        landmarks=landmarks[:,:3] 
        #landmarks=(landmarks+hips) 
        landmarks=sim.align_human_to_robot(landmarks)
        """ax.cla()
        coords = sim.get_coordinates()
        pn=[
                "right_wrist", "left_wrist",
                "right_ankle", "left_ankle",
                "right_elbow", "left_elbow",
                "right_knee", "left_knee"
            ]
        p = np.array([
            coords[k] for k in pn
        ])
        #ax=extractor.plot_world_landmarks(landmarks.copy(),ax,points=p,pointnames=pn)#sim.get_coords_of(["right_elbow", "left_elbow", "right_ankle","left_ankle"]))"""
        #get the hand and ankle links
        trajectories=sim.get_trajectories(["right_wrist", "left_wrist", "right_ankle", "left_ankle", "right_elbow", "left_elbow", "right_knee", "left_knee"],
                                        [landmarks[16],landmarks[15],landmarks[28],landmarks[27],landmarks[14],landmarks[13],landmarks[26],landmarks[25]])
        return trajectories,landmarks
    return None
if __name__ == "__main__":
    extractor = PoseExtractor(missing_value=-1.0)
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/home/dexter/.cache/kagglehub/datasets/nandwalritik/yoga-pose-videos-dataset/versions/2/Yoga_Vid_Collected/Abhay_Bhujangasana.mp4")
    #/home/dexter/Documents/GitHub/pose-to-biped/assets/walking.mp4
    #"/home/dexter/.cache/kagglehub/datasets/nandwalritik/yoga-pose-videos-dataset/versions/2/Yoga_Vid_Collected/Abhay_Bhujangasana.mp4"
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")
    sim = MujocoSimulator(
            "/home/dexter/Documents/GitHub/pose-to-biped/Robots/scene.xml"
        )
    
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer: 
        viewer.cam.distance = 5.0       
        ki_mod=kinematics_tranfser("/home/dexter/Documents/GitHub/pose-to-biped/Robots/h1_with_hand.urdf")
        for _ in range(5):
            sim.set_step(1)
            viewer.sync()

        for i in range(20):
            ret, frame = cap.read()
            if not ret:
                continue
            trajectories, landmarks = get_traj(extractor, sim, frame)
            sim.set_points([
                landmarks[16], landmarks[15],
                landmarks[28], landmarks[27],
                landmarks[12], landmarks[11]
            ])
            sim.set_step(1)  
            viewer.sync()
        sim.rotate_robot_to_human(landmarks)
        ki_mod.equalise_sims(sim)
        while viewer.is_running():
            ret, frame = cap.read()
            if not ret:
                break
            
            trajectories,landmarks=get_traj(extractor,sim,frame)
                #trajectories=[landmarks[14],landmarks[13],landmarks[28],landmarks[27]]
            if trajectories is not None:
                movements = ki_mod.move_to(
                                            ["right_hand_link", "left_hand_link", "right_ankle_link","left_ankle_link","right_elbow_link","left_elbow_link","right_knee_link","left_knee_link"],
                                            targets=np.array(trajectories),
                                            max_iter=20
                                        )
                for dic in movements:
                    sim.map_move(dic)
                    # Update MuJoCo kinematics
                    sim.set_step(1)     
                    sim.zero(dic)
                    viewer.sync()
            else: 
                pass #will need to reset 
            sim.set_points([landmarks[16],landmarks[15],landmarks[28],landmarks[27],landmarks[12],landmarks[11]])
            viewer.sync()
            #print(np.max(landmarks)-np.min(landmarks))
            ki_mod.equalise_sims(sim)
            """frame = cv2.resize(frame, (640, 480))  # match webcam frame
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

            img = img[..., :3]  # convert RGBA → RGB
            img = cv2.resize(img, (640, 480))
            frame = np.concatenate((frame,img), axis=1).astype(np.uint8)"""
            cv2.imshow("debug_frame", frame)
            #if cv2.waitKey(1) & 0xFF == 27:
                #break
             
    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
