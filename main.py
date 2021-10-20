"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""
import cv2
from HandPoseEstimation import PoseEstiamtion
from TVControl import TVControl

from HandDectection import HandDetector


def main():
    #class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
    #model = keras.models.load_model('handrecognition_model.h5')
    tv = TVControl()
    #tv.TurnOffTV()
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.75, trackCon=0.5)
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

            tv.DoAction(action)

        cv2.imshow("Image", img)

        k = cv2.waitKey(1)
        if k == 27:
            break



if __name__ == "__main__":
    main()

# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# Recreate the exact same model, including weights and optimizer.
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
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