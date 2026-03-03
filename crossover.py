from pose import PoseExtractor
from sim import MujocoSimulator 
import mujoco
import mujoco.viewer


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    extractor = PoseExtractor(missing_value=-1.0)
    cap = cv2.VideoCapture(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sim = MujocoSimulator(
            "C:/Users/dexte/Documents/mujoco_menagerie-main/mujoco_menagerie-main/unitree_h1/scene.xml"
        )
    j=0
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:        
        while viewer.is_running():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = extractor.process(frame)
            print(landmarks.shape)
            plt.cla()
            ax = extractor.plot_world_landmarks(landmarks, ax)
            sim.set_position(sim.initial+(j/10000))
            sim.set_step(10)
            viewer.sync()       
            plt.pause(0.005)
            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            j+=1
             
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()