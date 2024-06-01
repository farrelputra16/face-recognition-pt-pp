from flask import Flask, Response
from flask_cors import CORS
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import subprocess
import platform
from sklearn.neighbors import KNeighborsClassifier
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

def speak(str1):
    if platform.system() == 'Darwin':  # Check if the operating system is macOS
        subprocess.call(['say', str1])
    else:
        from win32com.client import Dispatch
        speak = Dispatch("SAPI.SpVoice")
        speak.Speak(str1)

video = cv2.VideoCapture(0)
if not video.isOpened():
    logging.error("Error: Could not open video.")
else:
    logging.debug("Video opened successfully.")

facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
if facedetect.empty():
    logging.error("Error: Could not load Haar cascade.")

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

logging.debug(f'Shape of Faces matrix --> {FACES.shape}')

# Extract IDs and names separately
ids = [label[0] for label in LABELS]
names = [label[1] for label in LABELS]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, ids)  # Fit the model using IDs

imgBackground = cv2.imread("background.png")
if imgBackground is None:
    logging.error("Error: Could not load background image.")

COL_NAMES = ['ID', 'NAME', 'TIME']

def generate_frames():
    while True:
        ret, frame = video.read()
        if not ret:
            logging.error("Error: Could not read frame from video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            predicted_id = knn.predict(resized_img)[0]  # Predict the ID
            name = names[ids.index(predicted_id)]  # Get the corresponding name
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, f"{predicted_id} - {name}", (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            attendance = [str(predicted_id), name, timestamp]

        frame_resized = cv2.resize(frame, (640, 480))
        imgBackground[162:162 + 480, 55:55 + 640] = frame_resized
        ret, buffer = cv2.imencode('.jpg', imgBackground)
        if not ret:
            logging.error("Error: Could not encode frame.")
        else:
            logging.debug("Frame encoded successfully, buffer size: %d", len(buffer))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
