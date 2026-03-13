from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("deepsafe_model.h5")

IMG_SIZE = 128

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"error": "No face detected"})

    x, y, w, h = faces[0]

    face = img[y:y+h, x:x+w]

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face)[0][0]

    result = "Fake" if prediction > 0.5 else "Real"

    return jsonify({
        "prediction": result,
        "confidence": float(prediction)
    })


if __name__ == "__main__":
    app.run(debug=True)