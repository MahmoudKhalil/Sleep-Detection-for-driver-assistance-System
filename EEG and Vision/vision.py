# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
from cv2 import cv2
import time


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.19
EYE_AR_CONSEC_FRAMES = 48
FRAME_DIFFERENCE_THRESH = 10
TIME_SLICE = 2000
IS_ASLEEP = 0
IS_DETECTED = False
outputQueue = []

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
# ConsecutiveFrames=0
leftCounter = 0
rightCounter = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
time.sleep(1.0)

# loop over frames from the video stream
while True:
    milli_sec = int(round(time.time() * 1000))
    endtime = milli_sec + TIME_SLICE
    leftEARAverage = 0
    rightEARAverage = 0
    leftWeights = 0
    rightWeights = 0
    rightCounter = 0
    leftCounter = 0
    while (int(round(time.time() * 1000)) < endtime):
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        ret, frame = vs.read()
        frame = imutils.resize(frame, width=450)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        if len(rects) != 0:
            IS_DETECTED = True
            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if leftEAR < EYE_AR_THRESH or rightEAR < EYE_AR_THRESH:
                    if leftEAR < EYE_AR_THRESH:
                        leftCounter += 1

                    if rightEAR < EYE_AR_THRESH:
                        rightCounter += 1
                else:
                    if leftEAR >= EYE_AR_THRESH:
                        if leftCounter > 0:
                            leftCounter -= 1

                    if rightEAR >= EYE_AR_THRESH:
                        if rightCounter > 0:
                            rightCounter -= 1

                leftEARAverage += leftEAR * (leftCounter + 1)
                leftWeights += leftCounter + 1

                rightEARAverage += rightEAR * (rightCounter + 1)
                rightWeights += (rightCounter + 1)

                # draw the computed eye aspect ratio on the frame to help
                # with debugging and setting the correct eye aspect ratio
                # thresholds and frame counters
                cv2.putText(frame, "L-EAR: {:.2f}".format(leftEAR), (150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "R-EAR: {:.2f}".format(rightEAR), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            IS_DETECTED = False

        # show the frame
        # COUNTER += 1
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q") or key == 27:
            break

    if not (IS_DETECTED == False and (rightWeights == 0 or leftWeights == 0)):
        rightEARAverage = rightEARAverage / rightWeights
        leftEARAverage = leftEARAverage / leftWeights
        # print("right EAR average = " + str(rightEARAverage))
        # print("left EAR average = " + str(leftEARAverage))
        if rightEARAverage < EYE_AR_THRESH and leftEARAverage < EYE_AR_THRESH:
            IS_ASLEEP = 1
        else:
            IS_ASLEEP = 0
        outputQueue.append((round(leftEARAverage, 4), round(rightEARAverage, 4), IS_ASLEEP))
        # outputQueue.append("{:.4f}".format(leftEARAverage) + " " + "{:.4f}".format(rightEARAverage) + " " + str(IS_ASLEEP) + "\n")
        # print("COUNTER = " + str(COUNTER))
        print(outputQueue.pop(0))
        COUNTER = 0
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or key == 27:
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()