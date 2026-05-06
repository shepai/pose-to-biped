import sys 
import os 
sys.path.append("/home/dexter/Documents/GitHub/pose-to-biped/") #replace with your file path
from pose import PoseExtractor, PARENTS
extractor = PoseExtractor(missing_value=-1.0)
import cv2
import numpy as np

def quickconvert(filepath):
     # Folder containing videos
    video_folder="/home/dexter/.cache/kagglehub/datasets/nandwalritik/yoga-pose-videos-dataset/versions/2/Yoga_Vid_Collected/"
    # Supported video extensions
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    # List all video files in folder
    video_files = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith(video_exts)
    ]
    print("Found videos:")
    for v in video_files:
        print(v)
    # Open videos one at a time
    X=[]
    c=0
    inds=[]
    for video_path in video_files:
        print(f"\nPlaying: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break   # End of video
            frame=frame[100:-100,100:-100]
            landmarks = extractor.process(frame)
            X.append(landmarks)
            c+=1
        inds.append(c)
        cap.release()
        np.save(filepath+"/X",np.array(X))
        np.save(filepath+"/IND",np.array(inds))

    cv2.destroyAllWindows()
    
quickconvert("/home/dexter/Documents/GitHub/pose-to-biped/models/")