from flask import Flask, render_template_string, request, jsonify, send_from_directory
import base64
import cv2
import numpy as np
import math
import time
import os

from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------- CONFIG ---------- #
DATA_DIR = "Data"           # same as training
MODEL_PATH = "sign_model.h5"
IMG_SIZE = 300
OFFSET = 20
CONFIDENCE_THRESHOLD = 0.70
CONFIRM_TIME = 2            # seconds
# ---------------------------- #

app = Flask(__name__)

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

# Rebuild labels from folder structure
datagen = ImageDataGenerator(rescale=1./255)
dummy_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
class_indices = dummy_gen.class_indices           # {'A':0, 'B':1, ...}
labels = {v: k for k, v in class_indices.items()} # {0:'A', 1:'B', ...}
print("Label mapping:", labels)

detector = HandDetector(maxHands=1)

# State for word building
current_letter = ""
word = ""
prediction_start = 0.0
last_add_time = 0.0

# --------- Serve static index.html --------- #
@app.route("/")
def index():
    # Serve the index.html from current folder
    return send_from_directory(".", "index.html")

# --------- Util: decode base64 frame --------- #
def decode_base64_image(data_url):
    """Convert base64 data URL from JS to OpenCV BGR image."""
    # remove "data:image/jpeg;base64," prefix
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    return img

# --------- /process_frame endpoint --------- #
@app.route("/process_frame", methods=["POST"])
def process_frame():
    global current_letter, word, prediction_start, last_add_time

    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"success": False, "error": "No frame received"}), 400

    try:
        img = decode_base64_image(data["frame"])
        # mirror (same as CSS transform scaleX(-1))
        img = cv2.flip(img, 1)

        img_rgb = img.copy()
        hands, img_annotated = detector.findHands(img_rgb)

        if not hands:
            return jsonify({"success": True, "letter": None, "confidence": 0.0, "word": word})

        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # clamp bounds
        y1 = max(0, y - OFFSET)
        y2 = min(img.shape[0], y + h + OFFSET)
        x1 = max(0, x - OFFSET)
        x2 = min(img.shape[1], x + w + OFFSET)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

        aspect = h / w
        try:
            if aspect > 1:
                s = IMG_SIZE / h
                wCal = math.ceil(s * w)
                imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                gap = math.ceil((IMG_SIZE - wCal) / 2)
                imgWhite[:, gap:gap + wCal] = imgResize
            else:
                s = IMG_SIZE / w
                hCal = math.ceil(s * h)
                imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                gap = math.ceil((IMG_SIZE - hCal) / 2)
                imgWhite[gap:gap + hCal, :] = imgResize
        except Exception as e:
            print("Resize error:", e)
            return jsonify({"success": True, "letter": None, "confidence": 0.0, "word": word})

        # prepare input
        input_img = imgWhite / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        prediction = model.predict(input_img, verbose=0)
        idx = int(np.argmax(prediction))
        letter = labels[idx]
        confidence = float(np.max(prediction))

        now = time.time()

        if confidence > CONFIDENCE_THRESHOLD:
            if letter != current_letter:
                current_letter = letter
                prediction_start = now

            if (now - prediction_start > CONFIRM_TIME) and (now - last_add_time > CONFIRM_TIME):
                word += letter
                last_add_time = now

        return jsonify({
            "success": True,
            "letter": current_letter,
            "confidence": confidence,
            "word": word
        })

    except Exception as e:
        print("Error in /process_frame:", e)
        return jsonify({"success": False, "error": str(e)}), 500

# --------- /get_text endpoint --------- #
@app.route("/get_text", methods=["GET"])
def get_text():
    return jsonify({"text": word})

# --------- /clear_text endpoint --------- #
@app.route("/clear_text", methods=["GET"])
def clear_text():
    global word, current_letter, prediction_start, last_add_time
    word = ""
    current_letter = ""
    prediction_start = 0.0
    last_add_time = 0.0
    return jsonify({"success": True})

if __name__ == "__main__":
    # run on localhost:5000
    app.run(debug=True)

