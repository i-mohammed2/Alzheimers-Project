import cv2
from ultralytics import YOLO
import pandas as pd

model = YOLO('yolov8m-pose.pt')

results = model(r"C:\Users\imguest.DESKTOP-6DE526B\Documents\Alzheimers Project\vid1_cam2_cropped.mp4")
results.show()

if results.keypoints is not None:
    print(results.keypoints)
    print(results.xyxy[0])
    print(results.xyn[0])
    print(results.conf[0])
    df = pd.DataFrame(results.keypoints, columns = ['xy', 'xyn', 'confidence'])
    df.to_csv('vid1_cam1.csv', index = False)

