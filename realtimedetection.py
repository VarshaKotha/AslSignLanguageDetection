import cv2
import numpy as np
import json
from keras._tf_keras.keras.models import model_from_json
import os
import sys
import io
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Apply encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the model
with open("2signlanguagedetectionmode48x48.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("2signlanguagedetectionmodel48x48.h5")
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)  # Wait for 2 seconds before starting
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

while True:
    ret, frame = cap.read()
    print("Capture status:", ret)
    if not ret:
        print("Frame not captured")
        break

    # Define ROI (Region of Interest) and process it
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    cropframe = frame[40:300, 0:300]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (48, 48))
    cropframe = extract_features(cropframe)
    # Make prediction
    pred = model.predict(cropframe, verbose=0)  # Reduce verbosity to improve performance
    prediction_label = label[np.argmax(pred)]

    # Overlay the prediction on the frame
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label} {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the output frame
    cv2.imshow("output", frame)

    # Check if the 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
