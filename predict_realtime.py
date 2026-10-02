import os
import time
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from asl_utils import FastPredictor, preprocess_hand_crop

# ---------------- CONFIGURATION ---------------- #
MODEL_PATH = "sign_model.h5"
LABELS_PATH = "labels.json"
DATA_DIR = "Data"

IMG_SIZE = 300
OFFSET = 20
CONFIDENCE_THRESHOLD = 0.80
MARGIN_THRESHOLD = 0.12
CONFIRM_TIME = 1.8  # Seconds gesture must remain steady
# ----------------------------------------------- #

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[predict_realtime] Error: Model file '{MODEL_PATH}' not found!")
        print("Please train the model first by running: python train_model.py")
        return

    print(f"[predict_realtime] Initializing model and hand detector...")
    predictor = FastPredictor(model_path=MODEL_PATH, labels_path=LABELS_PATH, data_dir=DATA_DIR)
    detector = HandDetector(maxHands=1, detectionCon=0.6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[predict_realtime] Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_letter = ""
    word = ""
    prediction_start = 0.0
    last_add_time = 0.0
    prev_frame_time = time.time()

    print("\n" + "=" * 50)
    print(" 🤟 ASL Real-Time Prediction App Loaded")
    print(" Controls: [Q] Quit | [C] Clear | [B] Backspace | [SPACE] Space | [S] Speak")
    print("=" * 50 + "\n")

    while True:
        success, img = cap.read()
        if not success:
            print("[predict_realtime] Failed to read frame from webcam.")
            break

        # Mirror frame
        img = cv2.flip(img, 1)

        # Detect hands
        hands, img = detector.findHands(img, draw=True)

        # Compute FPS
        curr_frame_time = time.time()
        fps = 1.0 / (curr_frame_time - prev_frame_time) if (curr_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = curr_frame_time

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            img_white = preprocess_hand_crop(img, hand, img_size=IMG_SIZE, offset=OFFSET)

            if img_white is not None:
                letter, confidence, margin, _ = predictor.predict(img_white)
                now = time.time()

                if confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
                    if letter != current_letter:
                        current_letter = letter
                        prediction_start = now

                    # Progress towards confirmation gauge
                    hold_duration = now - prediction_start
                    progress = min(1.0, hold_duration / CONFIRM_TIME)

                    if hold_duration >= CONFIRM_TIME and (now - last_add_time >= CONFIRM_TIME):
                        word += letter
                        last_add_time = now
                        print(f"Accepted letter: {letter} | Word: '{word}'")

                    # Draw confidence & bounding box graphics
                    color = (0, 255, 0) if progress >= 1.0 else (0, 215, 255)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

                    # Badge header
                    cv2.rectangle(img, (x, y - 40), (x + w, y), color, cv2.FILLED)
                    cv2.putText(img, f"{letter} ({confidence * 100:.1f}%)", (x + 8, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                    # Progress bar
                    bar_w = int(w * progress)
                    cv2.rectangle(img, (x, y + h + 10), (x + w, y + h + 22), (50, 50, 50), cv2.FILLED)
                    cv2.rectangle(img, (x, y + h + 10), (x + bar_w, y + h + 22), color, cv2.FILLED)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(img, "Low confidence", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Draw Bottom Information Strip
        img_h, img_w = img.shape[:2]
        strip = np.zeros((100, img_w, 3), dtype=np.uint8)
        
        cv2.putText(strip, f"FPS: {int(fps)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(strip, f"Word: '{word}'", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        combined_view = np.vstack((img, strip))
        cv2.imshow("ASL Vision Stream", combined_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word = ""
            print("Word cleared.")
        elif key == ord('b') or key == 8:  # Backspace key
            word = word[:-1] if len(word) > 0 else ""
            print(f"Backspace | Word: '{word}'")
        elif key == 32:  # Space key
            if word and not word.endswith(" "):
                word += " "
                print(f"Space added | Word: '{word}'")
        elif key == ord('s'):
            print(f"Speaking text: '{word}'")
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(word)
                engine.runAndWait()
            except Exception:
                pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
