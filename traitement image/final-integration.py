from imutils import face_utils
import numpy as np
import cv2

import math
import dlib
import os
import vlc

os.add_dll_directory(r'C:\Program Files\VideoLAN\VLC')


def euclideanDist(a, b):
    dist = math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))
    return (dist)

def EAR(eye): #EAR
    ear = (euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3]))
    return (ear)

def writeEyes(a, b, img):
    y1 = max(a[1][1], a[2][1])
    y2 = min(a[4][1], a[5][1])
    x1 = a[0][0]
    x2 = a[3][0]
    cv2.imwrite('images/left-eye.jpg', img[y1:y2, x1:x2])
    
    y1 = max(b[1][1], b[2][1])
    y2 = min(b[4][1], b[5][1])
    x1 = b[0][0]
    x2 = b[3][0]
    cv2.imwrite('images/right-eye.jpg', img[y1:y2, x1:x2])

def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10])+euclideanDist(mouth[4], mouth[8]))/(2*euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    image_points = np.array([
                                shape[33],    # Nose tip
                                shape[8],     # Chin
                                shape[45],    # Left eye left corner
                                shape[36],    # Right eye right corne
                                shape[54],    # Left Mouth corner
                                shape[48]     # Right mouth corner
                            ], dtype="double")
    
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ])
    
    # Camera internals
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return(translation_vector[1][0])



alert = vlc.MediaPlayer('alert/alert-sound.mp3')

frame_thresh_1 = 30
frame_thresh_2 = 30
frame_thresh_3 = 30
close_thresh = 0.18 #0.18 was determined to be the optimum EAR threshold in researches

flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1


capture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('training_data/shape_predictor_68_face_landmarks.dat')

(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print((leftStart, leftEnd))
print((rightStart, rightEnd))
print((mouthStart, mouthEnd))

while(True):
    ret, frame = capture.read()
    size = frame.shape
    rects = detector(frame, 0)
    if(len(rects)):
        shape = face_utils.shape_to_np(predictor(frame, rects[0]))

        leftEye = shape[leftStart:leftEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        leftEAR = EAR(leftEye)

        rightEye = shape[rightStart:rightEnd]
        rightEyeHull = cv2.convexHull(rightEye)
        rightEAR = EAR(rightEye) 

        avgEAR = (leftEAR+rightEAR)/2.0
        eyeContourColor = (255, 255, 255)

        if(yawn(shape[mouthStart:mouthEnd])>0.6):
            cv2.putText(frame, "Yawn Detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            yawn_countdown=1

        if(avgEAR<close_thresh):
            flag+=1
            eyeContourColor = (0,255,255)
            if(yawn_countdown and flag>=frame_thresh_3): #mouth an eye sleep
                eyeContourColor = (147, 20, 255)
                cv2.putText(frame, "Drowsy after yawn", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
            elif(flag>=frame_thresh_2 and getFaceDirection(shape, size)<0): #body posture and eye sleep
                eyeContourColor = (255, 0, 0)
                cv2.putText(frame, "Drowsy (Body Posture)", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
            elif(flag>=frame_thresh_1):
                eyeContourColor = (0, 0, 255)
                cv2.putText(frame, "Drowsy (Normal)", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
        elif(avgEAR>close_thresh and flag): #eyes are open
            print("Flag reseted to 0")
            alert.stop()
            yawn_countdown=0
            map_flag=1
            flag=0

        if(map_counter>=3):#implement send message to family
            map_flag=1
            map_counter=0
            vlc.MediaPlayer('alert/takeabreak.wav').play()

        cv2.drawContours(frame, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(frame, [rightEyeHull], -1, eyeContourColor, 2)
        
        writeEyes(leftEye, rightEye, frame)

    if(avgEAR>close_thresh):
        alert.stop()

    cv2.imshow('Driver', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
        
capture.release()
cv2.destroyAllWindows()
