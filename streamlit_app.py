import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 128

model = tf.keras.models.load_model("deepsafe_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("DeepSafe - AI Deepfake Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected")

    else:

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)[0][0]

        if prediction > 0.5:
            result = "Fake"
        else:
            result = "Real"

        st.image(image, caption="Uploaded Image")

        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {float(prediction):.2f}")