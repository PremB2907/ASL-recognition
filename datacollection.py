import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import time
import math

# USER SETTINGS
folder = "Data/A"         # Change folder per alphabet (A, B, C …)
images_to_capture = 300   # Stop after 300 images (safety)
offset = 20
imgSize = 300

# Setup
os.makedirs(folder, exist_ok=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Webcam not accessible")

detector = HandDetector(maxHands=1)
counter = 0

print("Press S to save | Q to quit")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # Clamp boundaries
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        aspectRatio = h / w

        try:
            if aspectRatio > 1:     # Height > Width
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) // 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:                   # Width >= Height
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) // 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

        except:
            continue

        cv2.imshow("Crop", imgCrop)
        cv2.imshow("White", imgWhite)

    cv2.putText(img, f"Saved: {counter}/{images_to_capture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Webcam", img)

    key = cv2.waitKey(1)
    if key == ord("s") and hands:
        if counter < images_to_capture:
            counter += 1
            cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
            print("Saved:", counter)
        else:
            print("Dataset full for this class.")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
