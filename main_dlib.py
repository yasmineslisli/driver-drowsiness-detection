import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
import vlc
import train as train

def euclideanDist(a, b):
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))
def ear(eye):
    return ((euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3])))

open_avg = train.getAvg()
close_avg = train.getAvg()

alert = vlc.MediaPlayer('alert-sound.mp3')
frame_thresh = 15
close_thresh = (close_avg+open_avg)/2.0
flag = 0

print(close_thresh)

capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while(True):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if(len(rects)):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        leftEAR = ear(leftEye) #Get the left eye aspect ratio
        rightEAR = ear(rightEye) #Get the right eye aspect ratio
        avgEAR = (leftEAR+rightEAR)/2.0
        # print(avgEAR)
        if(avgEAR<close_thresh):
            # print('Closed', avgEAR)
            flag+=1
            if(flag>=frame_thresh):
                alert.play()
        elif(avgEAR>close_thresh and flag):
            # print('Opened', avgEAR)
            alert.stop()
            flag=0
        cv2.drawContours(gray, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(gray, [rightEyeHull], -1, (0, 255, 0), 1)
    cv2.imshow('Driver', gray)
    if(cv2.waitKey(1)==27):
        break
        
capture.release()
cv2.destroyAllWindows()