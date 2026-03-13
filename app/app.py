from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("deepsafe_model.h5")

IMG_SIZE = 128

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = "Fake"
    else:
        result = "Real"

    return jsonify({
        "prediction": result,
        "confidence": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)