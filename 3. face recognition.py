import cv2
import numpy as np
import os
import sys
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

names = ['Prajwal', 'Dhanush', '3', '4', '5', '6', '7']   # IDs → names

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # LBPH: lower confidence = better match
        # Choose threshold
        threshold = 60   # Adjust between 50–80 if needed

        if confidence < threshold:
            name = names[id]
            text_conf = f"{round(100 - confidence)}%"
            break
        else:
            name = "Unknown"
            text_conf = f"{round(100 - confidence)}%"

        cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, text_conf, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow("camera", img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # ESC key
        break

print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
os.execv(sys.executable, ["python", "app1.py"])