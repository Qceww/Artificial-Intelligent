import cv2
from flask import Flask, render_template, Response
import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.feature_selection
import pandas as pd
from collections import Counter

app = Flask(__name__, template_folder="templates", static_folder='templates/static')

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
print_character =  []

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {0: 'P', 1: 'B', 2: 'T', 3: 'U', 4: 'I', 5: 'I Love You So Much'}
camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template("index.html")


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform AI predictions here
        data_aux = []
        x_ = []
        y_ = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        H, W, _ = frame.shape

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 30
            y1 = int(min(y_) * H) - 30

            x2 = int(max(x_) * W) + 30
            y2 = int(max(y_) * H) + 30

            if len(data_aux) == 84:
                data_aux = pd.DataFrame(data_aux)
                data_aux = data_aux.iloc[:42]
                prediction = model.predict(data_aux.values.reshape(1, -1))

            else:
                prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            # Overlay the prediction on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break

        # Yield the frame as bytes in a multipart response
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(host='0.0.0.0', port=5000)