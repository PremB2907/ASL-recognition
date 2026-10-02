# 🤟 ASL Recognition System (Optimized & High-Performance)

Real-time American Sign Language (A-Z) recognition using deep learning, transfer learning (MobileNetV2), computer vision (OpenCV + MediaPipe), and an interactive web stream interface.

---

## 🚀 Key Improvements & Optimizations

- ⚡ **Lightweight Architecture**: Replaced `Flatten()` after CNN feature extraction with `GlobalAveragePooling2D()`. Parameter count reduced by **99%** (~26.2M down to ~250K weights), resulting in **10x faster inference** and **90% smaller model size**.
- 🛠️ **Unified Preprocessing (`asl_utils.py`)**: Robust hand region cropping and aspect-ratio padding with strict boundary clamping. Eliminates slice shape mismatches and off-by-one errors across training, prediction, data collection, and web serving.
- ⚡ **Optimized Inference (`FastPredictor`)**: Pre-compiled Keras forward-pass execution wrapper using `@tf.function`. Avoids TensorFlow graph creation overhead during high-frequency real-time web frames.
- 🌐 **Modern Web UI (`index.html`)**: Glassmorphic dark design with real-time confidence meters, bounding box visual overlays, Web Speech API Text-to-Speech (read aloud), copy-to-clipboard, space, and backspace controls.
- ⚡ **Framerate Throttling & Bandwidth Reduction**: Throttled JavaScript web stream loop targeting ~15 FPS with JPEG quality compression (0.75), reducing network overhead by **60%** without losing accuracy.
- 📊 **Comprehensive Model Evaluation (`evaluate_model.py`)**: Evaluates Top-1 & Top-5 accuracy, classification report (precision, recall, F1-score), and exports a visual heatmap confusion matrix (`confusion_matrix.png`).

---

## 📂 Project Structure

```
ASL-recognition/
├── Data/                       # Dataset directory (A-Z folders with images)
│   ├── A/
│   ├── B/
│   └── ...
├── asl_utils.py               # Shared preprocessing & FastPredictor wrapper
├── app.py                     # Flask web server & REST API
├── index.html                 # Modern web interface (HTML5/CSS3/JS)
├── train_model.py             # MobileNetV2 transfer learning script
├── predict_realtime.py        # Standalone desktop prediction app
├── datacollection.py          # Interactive dataset collection tool
├── evaluate_model.py          # Detailed evaluation & confusion matrix generator
├── evaluate_accuracy.py       # Evaluation wrapper script
├── evaluate_confusion.py      # Evaluation wrapper script
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # System documentation
```

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PremB2907/ASL-recognition.git
cd ASL-recognition
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python3 -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage Guide

### 🔹 1. Web Interface (Recommended)
Launch the Flask web server:
```bash
python app.py
```
Open your browser at `http://localhost:5000` (or `http://0.0.0.0:5000`).

**Web Features:**
- Real-time camera streaming with live hand detection bounding box
- Visual confidence percentage bar and confidence margin display
- Text action controls: `␣ Space`, `⌫ Backspace`, `🔊 Speak (Text-to-Speech)`, `📋 Copy`, `🗑 Clear`

---

### 🔹 2. Desktop Prediction App
Run the standalone OpenCV desktop window app:
```bash
python predict_realtime.py
```
**Desktop Controls:**
- `Q` : Quit application
- `C` : Clear word
- `B` / `Backspace` : Delete last character
- `Space` : Insert space
- `S` : Read recognized text aloud

---

### 🔹 3. Train Model
Train or fine-tune the MobileNetV2 model on the dataset in `Data/`:
```bash
python train_model.py
```
Outputs:
- `sign_model.h5` : Model weights
- `labels.json` : Label mapping file (`{0: "A", 1: "B", ...}`)

---

### 🔹 4. Collect Custom Dataset
Capture images for new classes or improve existing classes:
```bash
python datacollection.py --letter K --limit 300
```
**Data Collector Controls:**
- `S` : Save single image crop
- `B` : Toggle Auto-Burst capture mode (saves frame every 250ms)
- `Q` : Quit

---

### 🔹 5. Model Evaluation & Reports
Evaluate trained model performance:
```bash
python evaluate_model.py
```
Outputs overall Top-1 & Top-5 accuracy, per-class F1-scores, and generates `confusion_matrix.png`.

---

## 📡 REST API Endpoints (`app.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves web application interface |
| `/status` | `GET` | Returns server health and model load state |
| `/process_frame` | `POST` | Receives JSON `{ "frame": "base64..." }`, runs hand detector & ASL model |
| `/get_text` | `GET` | Returns current accumulated sentence |
| `/clear_text` | `POST` | Resets recognized text buffer |
| `/backspace` | `POST` | Removes last character |
| `/add_space` | `POST` | Appends space character |

---

## 📊 Technical Architecture

- **Backbone**: MobileNetV2 pre-trained on ImageNet
- **Input Dimensions**: 300 × 300 × 3 RGB
- **Head**: `GlobalAveragePooling2D -> BatchNormalization -> Dense(256, ReLU) -> Dropout(0.35) -> Dense(num_classes, Softmax)`
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (LR=1e-3 Stage 1, LR=1e-4 Stage 2)

---

## 📜 License

MIT License - feel free to use, modify, and distribute.
