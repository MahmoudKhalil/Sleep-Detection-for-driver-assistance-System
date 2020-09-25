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
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import random
import pickle
import os
import pyedflib
import mne
import math
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import socket
import csv
import time


################# Reading the data file for PCA ###################


F = open("wake_data.txt","r")
s = F.read()

l=s.splitlines()

# r= l.
# print(r)

r = str.split(l[0], " ")
wake_data=np.zeros((l.__len__(),r.__len__()-1))
# print(float(wake_data[0,0]))

for i in range(l.__len__()):
    r = str.split(l[i], " ")
    # print(r)
    for j in range(r.__len__()-1):
        # print(r[j])
        wake_data[i][j]=float(r[j])


F = open("sleep_1_data.txt","r")
s = F.read()

l=s.splitlines()


sleep_1_data=np.zeros((l.__len__(),r.__len__()-1))

for i in range(l.__len__()):
    r = str.split(l[i], " ")
    for j in range(r.__len__()-1):
        sleep_1_data[i][j]=float(r[j])









EEG_data = [0]*1000
EAR = [0]*2

# Named fields according to Warren doc !
FIELDS = {"COUNTER": 0, "DATA-TYPE": 1, "AF3": 4, "F7": 5, "F3": 2, "FC5": 3, "T7": 6, "P7": 7, "01": 8,"02": 9,"P8": 10, "T8": 11, "FC6": 14, "F4": 15, "F8": 12, "AF4": 13, "DATALINE_1": 16, "DATALINE_2": 17}

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    # print(b,a)
    y = scipy.signal.lfilter(b, a, data)
    return y



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

def apply_MMD(current_epoch):

    sum=0
    for j in range(0,10):
        # print(j)
        current_sub_epoch = current_epoch[(j * 100):(100 * (j + 1))]
        y_min=np.min(current_sub_epoch)
        y_max=np.max(current_sub_epoch)

        x_min=np.where(current_sub_epoch==y_min)[0]+1
        x_max=np.where(current_sub_epoch==y_max)[0]+1

        x_diff = x_max - x_min
        y_diff = y_max - y_min

        x_sq = math.pow(x_diff,2)
        y_sq = math.pow(y_diff,2)

        total_distance = x_sq + y_sq

        sum+= math.sqrt(total_distance)



    return sum

def apply_esis(current_epoch,v):

    sum=0
    for i in range (0,current_epoch.__len__()):

        sum+=math.pow(current_epoch[i],2)*v

    return sum

def data2dic(data):
    field_list = data.split(b',')

    if len(field_list) > 17:
        # print(len(field_list))
        return {field:float(field_list[index]) for field, index in FIELDS.items()}
    else:

        return -1

def save_csv(data):
    field_list = data.split(b',')
    if len(field_list) > 17:
        with open ('mytest.csv','a',newline='') as f:
            thewriter = csv.writer(f)
            thewriter.writerow(field_list)



class Cykit_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        p1 = os
        p1.system('cmd /c "cd C:/Users/youss/OneDrive/Desktop/GP/CyKit-master/CyKit-master/Py3 && C:/Users/youss/AppData/Local/Programs/Python/Python37/python.exe ./CyKIT.py 127.0.0.1 54123 6 generic+noheader"')



class Reading_EEG_thread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)


    def run(self):
        FIRST_ELECTRODE="AF3"
        SECOND_ELECTRODE="FC5"

        TCP_IP = "127.0.0.1"
        TCP_PORT = 54123

        BUFFER_SIZE = 256



        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_IP, TCP_PORT))
        s.send(b"\r\n")
        # Local buffer to store parts of the messages
        buffer = b''

        # with open('mytest.csv', 'w', newline='') as f:
        #     thewriter = csv.writer(f)
        #     thewriter.writerow(
        #         ['COUNTER', 'DATA-TYPE', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', '01', '02', 'P8', 'T8', 'FC6', 'F4',
        #          'F8', 'AF4', 'DATALINE_1', 'DATALINE_2'])

        F = open("data.txt", "a")

        # If when when split by \r, \r was the last character of the message, we know that we have to remove \n from
        # the begining of the next message
        remove_newline = False
        while True:
            temp_data = [0]*1000
            counter = 0
            start_time = time.time()
            vision = Vision_thread()
            vision.start()
            while True:
                # We read a chunk

                data = s.recv(BUFFER_SIZE, socket.MSG_WAITALL)
                # If we have to remove \n at the begining
                if remove_newline:
                    data = data[1:]
                    remove_newline = False

                # Splitting the chunk into the end of the previous message and the begining of the next message
                msg_parts = data.split(b'\r')

                # If the second part ends with nothing when splitted we will have to remove \n next time
                if msg_parts[-1] == b'':
                    remove_newline = True
                    # Therefore the buffer for the next step is empty
                    n_buffer = b''
                else:
                    # otherwise we store the begining of the next message as the next buffer
                    n_buffer = msg_parts[-1][1:]

                # We interprete a whole message (begining from the previous step + the end
                fields = data2dic(buffer + msg_parts[0])
                # save_csv(buffer + msg_parts[0])
                # We setup the buffer for next step
                buffer = n_buffer

                # Print all channel
                # print(fields)
                # print(fields.values())
                # temp=fields.values()
                # str1=""
                # str1 = ''.join(str(str(e) + " ") for e in fields.values())
                # for i in range(temp.__len__()):
                #     str1+=str(temp[i])
                #     str1+=" "
                # str1+="\n"
                # print(str1)
                # str1=str1
                # str1 += "\n"
                # F.write(str1)
                temp_data[counter]=(fields[FIRST_ELECTRODE]-fields[SECOND_ELECTRODE])
                counter = (counter+1)

                if (counter == 1000):
                    print("--- seconds ---", (time.time() - start_time))
                    break
            global EEG_data
            EEG_data = temp_data[:]
            process = Processing_thread()
            process.start()




class Processing_thread(threading.Thread):
    global EEG_data
    global EAR
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        current_epoch=EEG_data
        frequency_domain_epoch = scipy.fft(current_epoch)
        # print(current_epoch[1])
        # print(frequency_domain_epoch[1])
        delta_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 0.00001, 4.0, 100)
        theta_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 4.0, 8.0, 100)
        alpha_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 8.0, 13.0, 100)
        beta_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 13.0, 22.0, 100)
        gamma_frequency_domain = butter_bandpass_filter(frequency_domain_epoch, 30.0, 45.0, 100)

        delta_time_domain = np.real(scipy.ifft(delta_frequency_domain))
        theta_time_domain = np.real(scipy.ifft(theta_frequency_domain))
        alpha_time_domain = np.real(scipy.ifft(alpha_frequency_domain))
        beta_time_domain = np.real(scipy.ifft(beta_frequency_domain))
        gamma_time_domain = np.real(scipy.ifft(gamma_frequency_domain))

        delta_MMD = apply_MMD(delta_time_domain)
        theta_MMD = apply_MMD(theta_time_domain)
        alpha_MMD = apply_MMD(alpha_time_domain)
        beta_MMD = apply_MMD(beta_time_domain)
        gamma_MMD = apply_MMD(gamma_time_domain)

        delta_v = 100 * ((4 + 0) / 2)
        theta_v = 100 * ((4 + 8) / 2)
        alpha_v = 100 * ((8 + 13) / 2)
        beta_v = 100 * ((13 + 22) / 2)
        gamma_v = 100 * ((30 + 45) / 2)

        delta_esis = apply_esis(delta_time_domain, delta_v)
        theta_esis = apply_esis(theta_time_domain, theta_v)
        alpha_esis = apply_esis(alpha_time_domain, alpha_v)
        beta_esis = apply_esis(beta_time_domain, beta_v)
        gamma_esis = apply_esis(gamma_time_domain, gamma_v)

        feature_array = [delta_MMD,theta_MMD,alpha_MMD,beta_MMD,gamma_MMD,delta_esis,theta_esis,alpha_esis,beta_esis,gamma_esis]
        # print(feature_array)
        feature_array=np.reshape(feature_array, (-1, 10))
        Before_PCA_Data = np.concatenate((feature_array,np.concatenate((wake_data, sleep_1_data),axis=0) ),axis=0 )
        # print("Henaaa",Before_PCA_Data[0])
        x = StandardScaler().fit_transform(Before_PCA_Data)
        # print("X shape : " ,  x.shape)
        pca = PCA(n_components=2)

        PCA_Data =pca.fit_transform(x)
        Input_PCA = PCA_Data[0]
        Final_data = np.concatenate((Input_PCA,EAR),axis=0)
        Final_data=np.reshape(Final_data,(1,-1))
        # print("ANA JOOOOEEE " , Final_data)
        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        PredictedOutcome = loaded_model.predict(Final_data)

        print("------------------- Result : " , PredictedOutcome)





EYE_AR_THRESH = 0.19
EYE_AR_CONSEC_FRAMES = 48
FRAME_DIFFERENCE_THRESH = 10
TIME_SLICE = 8000
IS_ASLEEP = 0
IS_DETECTED = False
outputQueue = []
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


vs = cv2.VideoCapture(0)

class Vision_thread(threading.Thread):


    def __init__(self):
        threading.Thread.__init__(self)
        # print("[INFO] loading facial landmark predictor...")
        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        # (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        # (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # start the video stream thread
        # print("[INFO] starting video stream thread...")
        # vs = VideoStream(src=0).start()

        time.sleep(1.0)
    def run(self):

        global EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, FRAME_DIFFERENCE_THRESH, TIME_SLICE, IS_ASLEEP, IS_DETECTED, outputQueue, detector, predictor, vs
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # EYE_AR_THRESH = 0.19
        # EYE_AR_CONSEC_FRAMES = 48
        # FRAME_DIFFERENCE_THRESH = 10
        # TIME_SLICE = 8000
        # IS_ASLEEP = 0
        # IS_DETECTED = False
        # outputQueue = []

        # initialize the frame counter as well as a boolean used to
        # indicate if the alarm is going off
        COUNTER = 0
        # ConsecutiveFrames=0
        leftCounter = 0
        rightCounter = 0
        ALARM_ON = False

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        # print("[INFO] loading facial landmark predictor...")
        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        #
        # # grab the indexes of the facial landmarks for the left and
        # # right eye, respectively
        # (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        # (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        #
        # # start the video stream thread
        # print("[INFO] starting video stream thread...")
        # # vs = VideoStream(src=0).start()
        # vs = cv2.VideoCapture(0)
        # time.sleep(1.0)

        # loop over frames from the video stream
        # while True:
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

                    if leftEAR < EYE_AR_THRESH:
                        leftCounter+=1
                    else:
                        if leftCounter > 0:
                            leftCounter-=1

                    if rightEAR < EYE_AR_THRESH:
                        rightCounter+=1
                    else:
                        if rightCounter > 0:
                            rightCounter-=1

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
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF
            #
            # # if the `q` key was pressed, break from the loop
            # if key == ord("q") or key == 27:
            #     break

        if not (IS_DETECTED == False and (rightWeights == 0 or leftWeights == 0)):
            rightEARAverage = rightEARAverage / rightWeights
            leftEARAverage = leftEARAverage / leftWeights
            # print("right EAR average = " + str(rightEARAverage))
            # print("left EAR average = " + str(leftEARAverage))
            # if rightEARAverage < EYE_AR_THRESH and leftEARAverage < EYE_AR_THRESH:
            #     IS_ASLEEP = 1
            # else:
            #     IS_ASLEEP = 0
            outputQueue.append((round(leftEARAverage, 4), round(rightEARAverage, 4)))
            # outputQueue.append("{:.4f}".format(leftEARAverage) + " " + "{:.4f}".format(rightEARAverage) + " " + str(IS_ASLEEP) + "\n")
            # print("COUNTER = " + str(COUNTER))
            # print(outputQueue.pop(0))
            COUNTER = 0
            # if the `q` key was pressed, break from the loop
            # if key == ord("q") or key == 27:
            #     break
        value = outputQueue.pop(0)
        global EAR
        EAR[0] = value[0]
        EAR[1] = value[1]

        # do a bit of cleanup
        # cv2.destroyAllWindows()
        # vs.release()


############# main #####################



cykit_thread = Cykit_thread()
cykit_thread.start()

EEG_thread = Reading_EEG_thread()
EEG_thread.start()



