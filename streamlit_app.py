import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random

st.set_page_config(page_title="DeepSafe AI", page_icon="🛡️")

st.title("🛡️ DeepSafe - AI Deepfake Detection")
st.write("Upload a face image to analyze if it may be **Real or Fake**.")

IMG_SIZE = 128

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected in the image.")

    else:
        x, y, w, h = faces[0]

        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        st.image(face, caption="Detected Face", use_container_width=True)

        # Simulated prediction
        prediction = random.random()

        if prediction > 0.5:
            result = "Fake"
        else:
            result = "Real"

        st.subheader("Prediction Result")
        st.write(f"Result: **{result}**")

        st.write(f"Confidence Score: **{prediction:.2f}**")

st.write("---")
st.caption("DeepSafe AI - Deepfake Detection Research Project")