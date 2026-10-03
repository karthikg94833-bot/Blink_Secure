# Mouse Movement Project

This project allows you to control your mouse using your eyes and facial gestures. It consists of two main parts: a face recognition system to unlock the application and an eye-tracking system to control the mouse.

## How it Works

The project uses a combination of computer vision and machine learning techniques to achieve its functionality. Here's a breakdown of the workflow:

1.  **Face Dataset Creation:** Before using the application, you need to create a dataset of your face. The `1. face dataset.py` script captures 30 images of your face and stores them in the `dataset` folder.

2.  **Face Recognition Training:** The `2. face training.py` script trains a face recognition model using the images from the dataset. The trained model is saved as `trainer/trainer.yml`.

3.  **Face Recognition and Application Launch:** The `3. face recognition.py` script uses the trained model to recognize your face. If your face is recognized, it launches the main application, `app1.py`.

4.  **Eye-Tracking Mouse Control:** The `app1.py` script is a Flask web application that provides a web interface with a 3x3 grid. It also runs a background thread that uses your computer's camera to track your eye movements.
    *   **Mouse Movement:** The application tracks your iris and moves the mouse cursor accordingly.
    *   **Mouse Click:** A blink is interpreted as a mouse click.

5.  **Grid-based Password:** The web interface allows you to set a password using the 3x3 grid. This provides an additional layer of security.

## Source Code

Here is the source code for the main components of the project:

### `1. face dataset.py`

```python
import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

#make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

#start detect your face and take 30 pictures
while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
```

### `2. face training.py`

```python
import cv2
import numpy as np
from PIL import Image #pillow package
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        print(imagePath)
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale

        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
```

### `3. face recognition.py`

```python
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
```

### `app1.py`

```python
import os

# ----------- SUPPRESS MEDIAPIPE + TF LITE WARNINGS -----------
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ---------------------------------------------------------------

import threading
import time
import signal

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pyautogui

import cv2
import mediapipe as mp

# ---------------- CONFIG ----------------
SMOOTHING = 0.20
BLINK_THRESHOLD = 0.004
BLINK_COOLDOWN = 0.9
MOVE_DURATION = 0.12
CAM_INDEX = 0
# ----------------------------------------

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

screen_w, screen_h = pyautogui.size()

grid_positions = {
    "1": (screen_w * 0.2, screen_h * 0.2),
    "2": (screen_w * 0.5, screen_h * 0.2),
    "3": (screen_w * 0.8, screen_h * 0.2),
    "4": (screen_w * 0.2, screen_h * 0.5),
    "5": (screen_w * 0.5, screen_h * 0.5),
    "6": (screen_w * 0.8, screen_h * 0.5),
    "7": (screen_w * 0.2, screen_h * 0.8),
    "8": (screen_w * 0.5, screen_h * 0.8),
    "9": (screen_w * 0.8, screen_h * 0.8),
}

stop_event = threading.Event()
eye_thread_started = threading.Event()

# ==============================================================
#                   EYE TRACKER THREAD
# ==============================================================

def eye_tracker_loop():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    print("Opening camera...")

    cap = None
    while cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            print("Camera not available. Retrying in 2 seconds…")
            time.sleep(2)

    print("Camera ready. Eye tracker active.")

    cur_x = screen_w // 2
    cur_y = screen_h // 2
    last_blink_time = 0

    while not stop_event.is_set():

        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, retrying…")
            time.sleep(0.2)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            time.sleep(0.01)
            continue

        try:
            landmarks = result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            # iris landmark (pointer)
            iris_lm = landmarks[474 + 1]
            x = int(iris_lm.x * w)
            y = int(iris_lm.y * h)

            # map to screen
            target_x = (x / w) * screen_w
            target_y = (y / h) * screen_h

            # smoothing
            cur_x += (target_x - cur_x) * SMOOTHING
            cur_y += (target_y - cur_y) * SMOOTHING

            # bounds
            margin = 30
            cur_x = max(margin, min(screen_w - margin, cur_x))
            cur_y = max(margin, min(screen_h - margin, cur_y))

            pyautogui.moveTo(cur_x, cur_y, duration=MOVE_DURATION)

            # -------------- blink detection --------------
            top = landmarks[145]
            bottom = landmarks[159]
            eye_gap = top.y - bottom.y

            now = time.time()
            if eye_gap < BLINK_THRESHOLD and (now - last_blink_time) > BLINK_COOLDOWN:
                pyautogui.click()
                last_blink_time = now
            # ----------------------------------------------

        except:
            pass

        time.sleep(0.01)

    cap.release()
    face_mesh.close()
    print("Eye tracker thread stopped.")

# ==============================================================
#                           FLASK ROUTES
# ==============================================================

@app.route("/")
def index():
    if 'password_set' not in session:
        return redirect(url_for('setup_password'))
    return render_template("index.html")

@app.route("/setup-password")
def setup_password():
    if 'password_set' in session:
        return redirect(url_for('index'))
    return render_template("setup_password.html", mode="setup")

@app.route("/confirm-password")
def confirm_password():
    if 'temp_password' not in session:
        return redirect(url_for('setup_password'))
    if 'password_set' in session:
        return redirect(url_for('index'))
    return render_template("setup_password.html", mode="confirm")

@app.route("/save-password", methods=["POST"])
def save_password():
    data = request.get_json(silent=True) or {}
    password_sequence = data.get("password", [])

    if not password_sequence:
        return jsonify({"status": "error", "msg": "No password provided"}), 400

    if len(password_sequence) < 4:
        return jsonify({"status": "error", "msg": "Password must be at least 4 digits"}), 400

    if 'temp_password' not in session:
        # First time - store temporary password
        session['temp_password'] = password_sequence
        return jsonify({"status": "ok", "redirect": url_for('confirm_password')})
    else:
        # Confirmation - check if passwords match
        if password_sequence == session['temp_password']:
            session['password'] = password_sequence
            session['password_set'] = True
            session.pop('temp_password', None)
            return jsonify({"status": "ok", "redirect": url_for('index')})
        else:
            return jsonify({"status": "error", "msg": "Passwords don't match"}), 400

@app.route("/reset-password")
def reset_password():
    session.pop('temp_password', None)
    session.pop('password', None)
    session.pop('password_set', None)
    return redirect(url_for('setup_password'))

@app.route("/move", methods=["POST"])
def move_mouse():
    if 'password_set' not in session:
        return jsonify({"status": "error", "msg": "Please set password first"}), 403

    data = request.get_json(silent=True) or {}
    grid = str(data.get("grid", "")).strip()

    if grid not in grid_positions:
        return jsonify({"status": "error", "msg": "Invalid grid"}), 400

    x, y = grid_positions[grid]
    click_after = bool(data.get("click", False))

    pyautogui.moveTo(x, y, duration=MOVE_DURATION)
    if click_after:
        pyautogui.click()

    return jsonify({"status": "ok", "grid": grid})

@app.route("/status")
def status():
    return jsonify({
        "eye_tracker_running": eye_thread_started.is_set(),
        "screen_width": screen_w,
        "screen_height": screen_h,
        "password_set": session.get('password_set', False)
    })

def start_eye_thread_once():
    if eye_thread_started.is_set():
        return
    t = threading.Thread(target=eye_tracker_loop, daemon=True)
    t.start()
    eye_thread_started.set()

def handle_signal(signum, frame):
    stop_event.set()
    time.sleep(0.3)
    os._exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Starting eye tracker thread...")
    start_eye_thread_once()

    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
```

## Machine Learning Models

The project uses the following machine learning models:

*   **Face Detection:** `haarcascade_frontalface_default.xml` (a pre-trained Haar Cascade model from OpenCV)
*   **Face Recognition:** `trainer/trainer.yml` (a custom-trained LBPH face recognition model)
*   **Eye Tracking:** `mediapipe.solutions.face_mesh` (a pre-trained model from Google's MediaPipe)

## Libraries Used

The project uses the following Python libraries:

*   **Flask:** A web framework for creating the web application.
*   **pyautogui:** A library for programmatically controlling the mouse and keyboard.
*   **OpenCV (cv2):** A library for computer vision tasks, such as camera access, image processing, and face recognition.
*   **mediapipe:** A library by Google for building multimodal ML pipelines.
*   **numpy:** A library for numerical operations.
*   **Pillow:** A library for image manipulation.

## How to Run the Project

1.  **Install the dependencies:**
    ```bash
    pip install Flask pyautogui opencv-python mediapipe numpy Pillow
    ```
2.  **Create a face dataset:**
    ```bash
    python "1. face dataset.py"
    ```
    Enter a user ID when prompted.
3.  **Train the face recognition model:**
    ```bash
    python "2. face training.py"
    ```
4.  **Run the application:**
    ```bash
    python "3. face recognition.py"
    ```
    The application will start after successful face recognition.

## Project Details

This project is a demonstration of how to build a hands-free computer interface using eye tracking. The face recognition part acts as a security measure before launching the main application. The grid-based password provides an additional layer of security within the application itself. The project is well-structured, with separate scripts for different functionalities. However, the connection between the face recognition 'login' and the main application is basic; it simply launches the app without passing any user-specific information.
