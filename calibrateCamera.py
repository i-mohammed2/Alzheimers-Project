import numpy as np
import cv2 as cv
import glob
import pickle
from ultralytics import YOLO

CHECKERBOARD = (6,8)
frameSize = (1920,1080)
MIN_POINTS = 50
RECORD = True

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((1, 6*8, 3), np.float32)
objp[0, :, :2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('images/*.png')

for fname in images:
    print(fname)
    img = cv.imread(fname)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

print(imgpoints)
print(objpoints)
cv.destroyAllWindows()

ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
pickle.dump((camera_matrix, dist), open("calibration.pk1", "wb"))
pickle.dump(camera_matrix, open("camera_matrix.pk1", "wb"))
pickle.dump(dist, open("dist.pk1", "wb"))
print(ret)
#calculate projectionMatrix
R = cv.Rodrigues(rvecs[0])[0]
t = tvecs[0]
Rt = np.concatenate([R,t], axis=-1) # [R|t]
P1 = np.matmul(camera_matrix,Rt) # A[R|t]
print(P1)

objpoints = []
imgpoints = []

images = glob.glob('imagesCam2/*.png')

for fname in images:
    print(fname)
    img = cv.imread(fname)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

print(imgpoints)
print(objpoints)
cv.destroyAllWindows()

ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
pickle.dump((camera_matrix, dist), open("calibration2.pk1", "wb"))
pickle.dump(camera_matrix, open("camera_matrix2.pk1", "wb"))
pickle.dump(dist, open("dist2.pk1", "wb"))
print(ret)
#calculate projectionMatrix
R = cv.Rodrigues(rvecs[0])[0]
t = tvecs[0]
Rt = np.concatenate([R,t], axis=-1) # [R|t]
P2 = np.matmul(camera_matrix,Rt) # A[R|t]
print(P2)

#triagulate points
model = YOLO('yolov8n-pose.pt')  # load an official model
model2 = YOLO('yolov8n-pose.pt')
#model = YOLO('path/to/best.pt')  # load a custom model
source = 'rtsp://192.168.1.186:554/stream/main'
source2 = 'rtsp://192.168.1.185:554/stream/main'

results = model.predict(source, show=True, stream=True)  # predict on video
results2 = model2.predict(source2, show = True, stream = True)

#triangulate points
cv.triangulatePoints(P1, P2, results.keypoints[0], results2.keypoints[0])

#Use projectionMatrix to calculate 3D point
# points_4d_hom = cv.triangulatePoints(P, P, imgpoints[0], imgpoints[1])
# points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
# points_3d = points_4d[:3, :].T
# print(points_3d)

# img = cv.imread('imagesCam2/img10.png')
# h, w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, (w,h), 1, (w,h))

# dst = cv.undistort(img, camera_matrix, dist)

# cv.imwrite('calibresult.png', dst)

# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error

# print("total error: {}".format(mean_error/len(objpoints)))