import numpy as np
import cv2 as cv
import glob
import pickle

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

img = cv.imread('images/img0.png')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, camera_matrix, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)))
