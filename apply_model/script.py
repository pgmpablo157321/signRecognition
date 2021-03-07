import tensorflow as tf
from tensorflow import keras

from scipy import stats

import cv2
import numpy as np

def preprocess_input(x, seed=42):
    x = tf.keras.applications.xception.preprocess_input(x)
    x = tf.image.resize(x, (299,299))
    return x

def mode(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]

vid = cv2.VideoCapture(0)

n_classes = 3
base_model = keras.applications.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
avg1 = keras.layers.Dense(40, activation="relu", kernel_regularizer=keras.regularizers.l1_l2(l1=1e-4, l2=1e-3))(avg)
output = keras.layers.Dense(n_classes, activation="softmax")(avg1)
model = keras.Model(inputs=base_model.input, outputs=output)
model.load_weights("modelo.h5")

arr = []
i = 0
steps = 20
while(True):
    # Capture the video frame by frame 
    ret, frame = vid.read() 
    # Display the resulting frame 
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    x = frame
    arr.append(x)
    if i%steps == steps-1:
        arr = np.array(arr)
        arr = preprocess_input(arr)
        ans = model.predict(arr, steps = 1)
        prediction = mode(np.argmax(ans, axis=1))
        if prediction == 0:
            print("Covid")
        if prediction == 1:
            print("Feliz")
        if prediction == 2:
            print("Manos")
        print(prediction)
        arr = []
    i+=1

vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 