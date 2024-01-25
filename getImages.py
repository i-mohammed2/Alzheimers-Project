import cv2
cam = cv2.VideoCapture('rtsp://192.168.1.185:554/stream/main')
cam2 = cv2.VideoCapture('rtsp://192.168.1.186:554/stream/main')

num = 0

while cam.isOpened():

    succes, img = cam.read()
    
    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('synchedImages/img' + str(num) + '.png', img)
        print("image saved!")
        

        

# Release and destroy all windows before termination
cam.release()

cv2.destroyAllWindows()