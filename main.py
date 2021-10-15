"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""
import cv2
import mediapipe as mp
import time
import os
from HandPoseEstimation import PoseEstiamtion
from TVControl import tvcontrol
# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# Recreate the exact same model, including weights and optimizer.
import numpy as np
#import matplotlib.pyplot as plt
import cv2
import pandas as pd
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList



def main():
    #class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
    #model = keras.models.load_model('handrecognition_model.h5')

    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionCon=0.75, trackCon=0.5)
    pe = PoseEstiamtion()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        Landmark = detector.findPosition(img)
        if len(Landmark) != 0:
            #print(pe.isThumbOpen(lmList))
            pe.GetPose(Landmark)
            pe.SwipeTick()
            action = pe.GetAction(Landmark)
            if not (action == "None"):
                print(action)

        cv2.imshow("Image", img)

        k = cv2.waitKey(1)
        if k == 27:
            break



if __name__ == "__main__":
    main()

# path = "./leapGestRecog/00/03_fist/frame_00_03_0001.png"
# X = []  # Image data
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (GRAY)
# img = cv2.resize(img, (320, 120))  # Reduce image size so training can be faster
#
# X.append(img)
# X = np.array(X, dtype="float32")
#
# X = X.reshape(1, 120, 320, 1)
# # test_image = test_image.reshape(img_width, img_height*3)    # Ambiguity!
# # Should this instead be: test_image.reshape(img_width, img_height, 3) ??
#
# result = model.predict(X)
# # print("HI")
# # print(np.argmax(result))
#
# # text = 'Predicted: {} {:2.0f}% '.format(class_names[np.argmax(result)])
# text = class_names[np.argmax(result)]
#
# img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)