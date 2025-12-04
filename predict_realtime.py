import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import math
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ---------------- #
DATA_DIR = "Data"           # Same folder as training
MODEL_PATH = "sign_model.h5"

IMG_SIZE = 300
OFFSET = 20

CONFIDENCE_THRESHOLD = 0.85   # required minimum confidence
MARGIN_THRESHOLD = 0.15       # top-1 vs top-2 prob difference
CONFIRM_TIME = 2.0            # seconds to keep same letter before locking
# ---------------------------------------- #


def build_label_mapping():
    """
    Rebuild class_indices from the same directory structure used during training.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    dummy_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    class_indices = dummy_gen.class_indices  # {'A': 0, 'B': 1, ...}
    labels = {v: k for k, v in class_indices.items()}  # {0: 'A', 1: 'B', ...}
    print("Class mapping (index -> label):", labels)
    return labels


def prepare_hand_image(img, hand_bbox):
    """
    Take original BGR frame + bbox dict from cvzone and return
    centered 300x300 RGB image on white background suitable for model input.
    """
    x, y, w, h = hand_bbox["bbox"]

    # Clamp crop bounds so we don't go out of frame
    y1 = max(0, y - OFFSET)
    y2 = min(img.shape[0], y + h + OFFSET)
    x1 = max(0, x - OFFSET)
    x2 = min(img.shape[1], x + w + OFFSET)

    img_crop = img[y1:y2, x1:x2]
    img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

    aspect = h / w if w != 0 else 1.0

    try:
        if aspect > 1:  # tall
            scale = IMG_SIZE / h
            w_cal = math.ceil(scale * w)
            img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
            gap = math.ceil((IMG_SIZE - w_cal) / 2)
            img_white[:, gap:gap + w_cal] = img_resize
        else:  # wide
            scale = IMG_SIZE / w if w != 0 else 1.0
            h_cal = math.ceil(scale * h)
            img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
            gap = math.ceil((IMG_SIZE - h_cal) / 2)
            img_white[gap:gap + h_cal, :] = img_resize
    except Exception as e:
        print("Resize error:", e)
        return None

    return img_white


def main():
    # --------- Sanity prints --------- #
    print("Working directory:", os.getcwd())
    print("Files here:", os.listdir())

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # --------- Load model --------- #
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    # --------- Build labels from DATA_DIR --------- #
    labels = build_label_mapping()

    # --------- Setup camera & detector --------- #
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Webcam not accessible")

    detector = HandDetector(maxHands=1)

    current_letter = ""      # what we are currently “tracking”
    word = ""                # final word being built
    prediction_start = 0.0   # when current_letter started being stable
    last_add_time = 0.0      # last time a letter was added

    print("Controls: q = Quit | c = Clear word | b = Backspace")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame")
            break

        # Flip horizontally for “mirror” feel
        img = cv2.flip(img, 1)

        # Detect hands
        hands, img = detector.findHands(img)  # detector also draws annotations

        # Display area for word on separate white strip
        word_strip = np.ones((100, 800, 3), np.uint8) * 255

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            processed = prepare_hand_image(img, hand)
            if processed is not None:
                # Prepare model input
                input_img = processed.astype("float32") / 255.0
                input_img = np.expand_dims(input_img, axis=0)

                # Model prediction
                prediction = model.predict(input_img, verbose=0)
                probs = prediction[0]
                idx = int(np.argmax(probs))
                letter = labels[idx]
                confidence = float(probs[idx])

                # Top-2 margin to avoid confused predictions
                sorted_idx = np.argsort(probs)[::-1]
                top1, top2 = sorted_idx[0], sorted_idx[1]
                margin = probs[top1] - probs[top2]

                now = time.time()

                # Only react if model is clearly confident AND margin is good
                if confidence > CONFIDENCE_THRESHOLD and margin > MARGIN_THRESHOLD:
                    if letter != current_letter:
                        current_letter = letter
                        prediction_start = now

                    # Letter accepted if it stayed same long enough + cooldown
                    if (now - prediction_start > CONFIRM_TIME) and (now - last_add_time > CONFIRM_TIME):
                        word += letter
                        last_add_time = now

                # Draw bounding box & prediction text
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(img, f"{letter} ({confidence:.2f})",
                            (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 0, 0), 2)
                cv2.putText(img, f"margin: {margin:.2f}",
                            (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

        # Show word strip
        cv2.putText(word_strip, f"Word: {word}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        cv2.imshow("Prediction", word_strip)
        cv2.imshow("Webcam", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word = ""
        elif key == ord('b'):
            word = word[:-1] if len(word) > 0 else ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
