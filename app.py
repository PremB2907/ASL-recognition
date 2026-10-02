import os
import base64
import time
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from cvzone.HandTrackingModule import HandDetector

from asl_utils import preprocess_hand_crop, load_label_mapping, FastPredictor

# ---------------- CONFIGURATION ---------------- #
MODEL_PATH = "sign_model.h5"
LABELS_PATH = "labels.json"
DATA_DIR = "Data"
IMG_SIZE = 300
OFFSET = 20
CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.10
CONFIRM_TIME = 1.8  # seconds hand sign must stay stable to add letter
# ----------------------------------------------- #

app = Flask(__name__)

# Global instances & state
predictor = None
detector = None

current_letter = ""
word = ""
prediction_start = 0.0
last_add_time = 0.0

def init_services():
    global predictor, detector
    if detector is None:
        try:
            detector = HandDetector(maxHands=1, detectionCon=0.6)
            print("[app] HandDetector initialized.")
        except Exception as err:
            print(f"[app] Error initializing HandDetector: {err}")

    if predictor is None and os.path.exists(MODEL_PATH):
        try:
            predictor = FastPredictor(model_path=MODEL_PATH, labels_path=LABELS_PATH, data_dir=DATA_DIR)
            print("[app] FastPredictor loaded successfully.")
        except Exception as err:
            print(f"[app] Warning: Failed to load FastPredictor ({err})")

def decode_base64_image(data_url):
    """Convert JS base64 image data string into OpenCV BGR numpy matrix."""
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    """Serve main ASL Vision interface."""
    return send_from_directory(".", "index.html")


@app.route("/status", methods=["GET"])
def status():
    """Return backend status and model metadata."""
    model_ready = predictor is not None and os.path.exists(MODEL_PATH)
    labels = predictor.idx_to_label if predictor else {}
    return jsonify({
        "status": "online",
        "model_loaded": model_ready,
        "model_path": MODEL_PATH,
        "classes_count": len(labels),
        "word": word
    })


@app.route("/process_frame", methods=["POST"])
def process_frame():
    """Receive camera frame from frontend, run hand detector & ASL model, update state."""
    global current_letter, word, prediction_start, last_add_time, predictor

    init_services()

    if predictor is None:
        return jsonify({
            "success": False,
            "error": "Model file not found or failed to load. Please train model first using train_model.py"
        }), 400

    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"success": False, "error": "No frame received"}), 400

    try:
        img = decode_base64_image(data["frame"])
        if img is None or img.size == 0:
            return jsonify({"success": False, "error": "Invalid frame data"}), 400

        # Mirror frame to match webcam user view
        img = cv2.flip(img, 1)

        hands, _ = detector.findHands(img, draw=False)
        if not hands:
            current_letter = ""
            return jsonify({
                "success": True,
                "hand_detected": False,
                "letter": None,
                "confidence": 0.0,
                "margin": 0.0,
                "word": word
            })

        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # Preprocess hand region into standardized 300x300 image
        img_white = preprocess_hand_crop(img, hand, img_size=IMG_SIZE, offset=OFFSET)
        if img_white is None:
            return jsonify({
                "success": True,
                "hand_detected": True,
                "letter": None,
                "confidence": 0.0,
                "margin": 0.0,
                "word": word
            })

        # Run optimized model prediction
        letter, confidence, margin, _ = predictor.predict(img_white)
        now = time.time()

        if confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
            if letter != current_letter:
                current_letter = letter
                prediction_start = now

            if (now - prediction_start >= CONFIRM_TIME) and (now - last_add_time >= CONFIRM_TIME):
                word += letter
                last_add_time = now

        return jsonify({
            "success": True,
            "hand_detected": True,
            "letter": letter if confidence >= CONFIDENCE_THRESHOLD else None,
            "confidence": confidence,
            "margin": margin,
            "word": word,
            "bbox": [x, y, w, h]
        })

    except Exception as err:
        print(f"[app] Error in /process_frame: {err}")
        return jsonify({"success": False, "error": str(err)}), 500


@app.route("/get_text", methods=["GET"])
def get_text():
    """Fetch current recognized sentence/word."""
    return jsonify({"text": word})


@app.route("/clear_text", methods=["GET", "POST"])
def clear_text():
    """Clear recognized text."""
    global word, current_letter, prediction_start, last_add_time
    word = ""
    current_letter = ""
    prediction_start = 0.0
    last_add_time = 0.0
    return jsonify({"success": True, "word": word})


@app.route("/backspace", methods=["POST"])
def backspace():
    """Remove last character from recognized text."""
    global word
    word = word[:-1] if len(word) > 0 else ""
    return jsonify({"success": True, "word": word})


@app.route("/add_space", methods=["POST"])
def add_space():
    """Add space character to recognized text."""
    global word
    if word and not word.endswith(" "):
        word += " "
    return jsonify({"success": True, "word": word})


if __name__ == "__main__":
    init_services()
    app.run(host="0.0.0.0", port=5000, debug=True)
