import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Recreate the exact same model, including weights and optimizer.
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tensorflow.keras.preprocessing import image
# Sklearn
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
# We need to get all the paths for the images to later load them
# imagepaths = []
#
# # Go through all the files and subdirectories inside a folder and save path to images inside list
# for root, dirs, files in os.walk(".", topdown=False):
#   for name in files:
#     path = os.path.join(root, name)
#     if path.endswith("png"): # We want only the images
#       imagepaths.append(path)
#
# print(len(imagepaths)) # If > 0, then a PNG image was loaded
# X = []  # Image data
# y = []  # Labels
#
# # Loops through imagepaths to load images and labels into arrays
# for path in imagepaths:
#   img = cv2.imread(path)  # Reads image and returns np.array
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (GRAY)
#   img = cv2.resize(img, (320, 120))  # Reduce image size so training can be faster
#   X.append(img)
#
#   # Processing label in image path
#   category = path.split("\\")[3]
#   label = int(category.split("_")[0][1])  # We need to convert 10_down to 00_down, or else it crashes
#   y.append(label)
#
# # Turn X and y into np.array to speed up train_test_split
# X = np.array(X, dtype="uint8")
# X = X.reshape(len(imagepaths), 120, 320, 1)  # Needed to reshape so CNN knows it's different images
# y = np.array(y)
#
# print("Images loaded: ", len(X))
# print("Labels loaded: ", len(y))
#
# print(y[0], imagepaths[0])  # Debugging
#
# ts = 0.3 # Percentage of images that we want to use for testing. The rest is used for training.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
#



model = keras.models.load_model('handrecognition_model.h5')
model.summary()
img_width, img_height = 312, 220
path = "./leapGestRecog/00/03_fist/frame_00_03_0001.png"
X = [] # Image data

img = cv2.imread(path) # Reads image and returns np.array
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
img = cv2.resize(img, (320, 120)) # Reduce image size so training can be faster

X.append(img)
X = np.array(X, dtype="uint8")

X = X.reshape(1, 120, 320, 1)
#test_image = test_image.reshape(img_width, img_height*3)    # Ambiguity!
# Should this instead be: test_image.reshape(img_width, img_height, 3) ??

result = model.predict(X)
print("HI")
print (np.argmax(result))
#test_loss, test_acc = model.evaluate(X_test, y_test)

# print('Test accuracy: {:2.2f}%'.format(test_acc*100))
#
# predictions = model.predict(X_test) # Make predictions towards the test set
#
# np.argmax(predictions[0]), y_test[0] # If same, got it right
#
# # img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img2 = image.load_img("C:/Users/willi/PycharmProjects/HandRecognition/leapGestRecog/00/01_palm/frame_00_01_0001.png",
#                       target_size=(320, 120))
# img_arr = image.img_to_array(img2)
# img_batch = np.expand_dims(img_arr, axis=0)
# img_preprocessed = preprocess_input(img_batch)
# pred = model.predict(img_preprocessed)
# print(decode_predictions(pred, top=3)[0])
#
#
#
# # Function to plot images and labels for validation purposes
# def validate_9_images(predictions_array, true_label_array, img_array):
#   # Array for pretty printing and then figure size
#   class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
#   plt.figure(figsize=(15, 5))
#
#   for i in range(1, 10):
#     # Just assigning variables
#     prediction = predictions_array[i]
#     true_label = true_label_array[i]
#     img = img_array[i]
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
#     # Plot in a good way
#     plt.subplot(3, 3, i)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(prediction)  # Get index of the predicted label from prediction
#
#     # Change color of title based on good prediction or not
#     if predicted_label == true_label:
#       color = 'blue'
#     else:
#       color = 'red'
#
#     plt.xlabel("Predicted: {} {:2.0f}% (True: {})".format(class_names[predicted_label],
#                                                           100 * np.max(prediction),
#                                                           class_names[true_label]),
#                color=color)
#   plt.show()

#validate_9_images(predictions, y_test, X_test)

