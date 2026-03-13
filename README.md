# 🛡️ DeepSafe – AI Deepfake Detection System

DeepSafe is an AI-powered system designed to detect manipulated or AI-generated facial media.
The project uses computer vision techniques and a machine learning pipeline to analyze faces and predict whether they are **Real or Fake**.

---

## 🚀 Features

* Face detection using **OpenCV Haar Cascades**
* Deepfake classification pipeline
* Image preprocessing for ML models
* Interactive **Streamlit web interface**
* Upload image → Detect face → Predict Real/Fake

---

## 🧠 Project Architecture

Image Upload
↓
Face Detection (OpenCV)
↓
Face Cropping & Preprocessing
↓
Deepfake Detection Model
↓
Prediction Output (Real / Fake)

---

## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy
* Streamlit
* scikit-learn
* Matplotlib

---

## 📂 Project Structure

DeepSafe-AI-Fake-Detection
│
├── app/
│   └── app.py
│
├── model/
│   └── train_model.py
│
├── utils/
│   └── preprocess.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ Installation

Clone the repository

git clone https://github.com/GarvitPro/DeepSafe-AI-Fake-Detection.git

Navigate into the project folder

cd DeepSafe-AI-Fake-Detection

Create a virtual environment

python -m venv venv

Activate the environment

Windows
venv\Scripts\activate

Install dependencies

pip install -r requirements.txt

---

## ▶️ Run the Application

Run the Streamlit web interface:

streamlit run streamlit_app.py

Then open the browser at:

http://localhost:8501

Upload an image and the system will analyze the face.

---

## 📊 Future Improvements

* Real deepfake detection CNN model
* Video deepfake detection
* Face heatmap visualization
* Larger deepfake datasets

---

## 👨‍💻 Author

Garvit Arora
B.Tech Computer Science & Business Systems
SRM Institute of Science and Technology

GitHub: https://github.com/GarvitPro
