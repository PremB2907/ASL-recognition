import os
import sys
import time
import cv2
import numpy as np
import argparse
from cvzone.HandTrackingModule import HandDetector
from asl_utils import preprocess_hand_crop

# ---------------- CONFIGURATION ---------------- #
DEFAULT_TARGET = "A"
DEFAULT_LIMIT = 300
IMG_SIZE = 300
OFFSET = 20
DATA_DIR = "Data"
# ----------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="ASL Dataset Collection Tool")
    parser.add_argument("--letter", type=str, default=None, help="Target letter folder (e.g. A, B, C...)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Target number of images to capture")
    args = parser.parse_args()

    letter = args.letter
    if not letter:
        input_val = input(f"Enter target letter/class name to collect (default '{DEFAULT_TARGET}'): ").strip().upper()
        letter = input_val if input_val else DEFAULT_TARGET

    target_folder = os.path.join(DATA_DIR, letter)
    os.makedirs(target_folder, exist_ok=True)

    # Count existing images in target folder
    existing_files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    counter = len(existing_files)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[datacollection] Error: Webcam not accessible.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(maxHands=1, detectionCon=0.6)
    auto_burst = False
    last_burst_time = 0.0

    print("\n" + "=" * 50)
    print(f" 📸 ASL Data Collection Tool | Target Class: '{letter}'")
    print(f" Saving to: {target_folder}/")
    print(" Controls: [S] Save 1 Image | [B] Toggle Auto-Burst | [Q] Quit")
    print("=" * 50 + "\n")

    while True:
        success, img = cap.read()
        if not success:
            print("[datacollection] Failed to capture frame.")
            break

        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, draw=True)

        img_white = None
        if hands:
            hand = hands[0]
            img_white = preprocess_hand_crop(img, hand, img_size=IMG_SIZE, offset=OFFSET)

            if img_white is not None:
                cv2.imshow("Hand Crop Preview (Input to Model)", img_white)

        now = time.time()
        if auto_burst and hands and img_white is not None and counter < args.limit:
            if now - last_burst_time >= 0.25:  # capture every 250ms
                counter += 1
                img_path = os.path.join(target_folder, f"Image_{time.time():.4f}.jpg")
                cv2.imwrite(img_path, img_white)
                last_burst_time = now
                print(f"Auto-saved: {counter}/{args.limit} -> {img_path}")

        # Status text overlay
        status_color = (0, 255, 0) if counter < args.limit else (0, 0, 255)
        mode_text = "[AUTO-BURST ACTIVE]" if auto_burst else "[MANUAL SAVE]"
        cv2.putText(img, f"Class: {letter} | Count: {counter}/{args.limit} {mode_text}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("ASL Data Collector", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and hands and img_white is not None:
            if counter < args.limit:
                counter += 1
                img_path = os.path.join(target_folder, f"Image_{time.time():.4f}.jpg")
                cv2.imwrite(img_path, img_white)
                print(f"Saved: {counter}/{args.limit} -> {img_path}")
            else:
                print(f"[datacollection] Target count limit reached ({args.limit}).")
        elif key == ord('b'):
            auto_burst = not auto_burst
            print(f"[datacollection] Auto-burst mode: {'ON' if auto_burst else 'OFF'}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[datacollection] Completed. Total images in '{letter}': {counter}")

if __name__ == "__main__":
    main()
