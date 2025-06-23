import cv2
import cv2.data
import tensorflow as tf
from keras.models import load_model
import numpy as np
import streamlit as st
from flask import Flask, render_template, request
import os

app = Flask(__name__)
model = load_model('ml_model/fmd_model.keras')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_annotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)[0]
        predicted_class = np.argmax(prediction)

        label = "MASK" if predicted_class == 1 else "NO MASK"
        color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)

        cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

    return image

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        img_path = os.path.join('static', 'result.png')

        # Case-1 Upload Image
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            file.save(img_path)
            image = cv2.imread(img_path)
            result_img = detect_annotate(image)
            cv2.imwrite(img_path, result_img)

            return render_template('index.html', result=True, img_path=img_path)

        # Case-2 Use Webcam
        elif 'use_webcam' in request.form:
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                cv2.imshow("Capture - Press SPACE", frame)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break
            cap.release()
            cv2.destroyAllWindows()
            if ret:
                result_img = detect_annotate(frame)
                cv2.imwrite(img_path, result_img)
            return render_template('index.html', result=True, img_path=img_path)
            
    return render_template('index.html', result=False)

if __name__ == '__main__':
    app.run(debug=True)