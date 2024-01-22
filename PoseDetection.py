from ultralytics import YOLO
import cv2 as cv
from cv2 import calibrateCamera
import numpy as np


CHECKERBOARD = (5,7)
MIN_POINTS = 50
RECORD = True
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((1, 5*7, 3), np.float32)
objp[0, :, :2] = np.mgrid[0:5,0:7].T.reshape(-1,2)

objpoints = []
imgpoints = []

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
model2 = YOLO('yolov8n-pose.pt')
#model = YOLO('path/to/best.pt')  # load a custom model
source = 'rtsp://192.168.1.186:554/stream/main'
source2 = 'rtsp://192.168.1.185:554/stream/main'
# Predict with the model
#results = model()  # predict on an image

results = model.predict(source, show=True, stream=True)  # predict on video
results2 = model2.predict(source2, show = True, stream = True)

camera_matrix1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
camera_matrix2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])


for r, r2 in zip(results, results2):
    # Get the center point of the bounding box from both models
    center_point1 = [r.boxes[0] + r.boxes[2] / 2, r.boxes[1] + r.boxes[3] / 2]
    center_point2 = [r2.boxes[0] + r2.boxes[2] / 2, r2.boxes[1] + r2.boxes[3] / 2]

    # Project the 2D points back to 3D space
    points1 = cv.undistortPoints(np.array([center_point1]), camera_matrix1, dist_coeffs1)
    points2 = cv.undistortPoints(np.array([center_point2]), camera_matrix2, dist_coeffs2)

    # Triangulate the points to find the 3D location
    point_4d_hom = cv.triangulatePoints(projMatr1, projMatr2, points1, points2)
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_4d[:3, :].T

    print(point_3d)