# 🤟 ASL Recognition System

A real-time **American Sign Language (ASL) Recognition System** built using **Deep Learning** and **Computer Vision**. This project focuses on recognizing hand gestures corresponding to ASL alphabets and converting them into meaningful textual output, enabling better communication accessibility.

---

## 🚀 Project Overview

The ASL Recognition System captures hand gestures through a webcam, processes the visual input using computer vision techniques, and classifies the gesture using a trained deep learning model. The system is designed to work in **real time**, making it suitable for interactive applications.

This project is aimed at:

* Bridging the communication gap between hearing-impaired and non-sign language users
* Demonstrating the practical application of CNNs and transfer learning
* Exploring real-time prediction using live video streams

---

## 🧠 Key Features

* 📷 Real-time hand gesture detection using webcam
* 🔤 Recognition of ASL alphabets (A–Z)
* 🧠 Deep Learning-based classification model
* ⚡ Fast and interactive predictions
* 📊 Scalable design for adding more gestures or words

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * TensorFlow / Keras
  * OpenCV
  * NumPy
  * MediaPipe
* **Model Type:** Convolutional Neural Network (CNN)
* **Environment:** Local system / Google Colab (for training)

---

## 📂 Project Structure

```
ASL-recognition/
│
├── dataset/                 # ASL image dataset (train/test)
├── model/                   # Trained model files
├── train_model.py            # Script to train the model
├── predict_realtime.py       # Real-time prediction using webcam
├── utils.py                  # Helper functions
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/PremB2907/ASL-recognition.git
cd ASL-recognition
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Train the Model

```bash
python train_model.py
```

### 🔹 Run Real-Time Prediction

```bash
python predict_realtime.py
```

Make sure your **webcam is connected** before running real-time prediction.

---

## 📈 Results & Performance

* The model demonstrates strong real-time prediction capability
* Accuracy may vary depending on lighting conditions and hand positioning
* Performance improves with a larger and more diverse dataset

---

## 🌱 Future Improvements

* Add word and sentence-level recognition
* Improve accuracy using transfer learning (MobileNet / EfficientNet)
* Deploy as a web or mobile application
* Add voice output for recognized signs

---

## 🤝 Contributors

* **Prem Sudesh Baraskar** – Project Lead & Developer

---

## 📜 License

This project is intended for **educational and research purposes**.

---


⭐ If you find this project useful, don’t forget to **star the repository** and share it with others interested in accessibility and AI!
