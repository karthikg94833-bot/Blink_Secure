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
    # Check if the user is currently locked out
    if 'lockout_until' in session and time.time() < session['lockout_until']:
        remaining_time = int(session['lockout_until'] - time.time())
        return jsonify({"status": "error", "msg": f"Too many failed attempts. Please wait {remaining_time // 60}m {remaining_time % 60}s."}), 429

    data = request.get_json(silent=True) or {}
    password_sequence = data.get("password", [])

    if not password_sequence:
        return jsonify({"status": "error", "msg": "No password provided"}), 400

    if len(password_sequence) < 4:
        return jsonify({"status": "error", "msg": "Password must be at least 4 digits"}), 400

    if 'temp_password' not in session:
        # First time - store temporary password
        session['temp_password'] = password_sequence
        # Initialize failure tracking
        session['login_attempts'] = 0
        return jsonify({"status": "ok", "redirect": url_for('confirm_password')})
    else:
        # Confirmation - check if passwords match
        if password_sequence == session['temp_password']:
            # On success, set the real password and clear tracking variables
            session['password'] = password_sequence
            session['password_set'] = True
            session.pop('temp_password', None)
            session.pop('login_attempts', None)
            session.pop('lockout_until', None)
            return jsonify({"status": "ok", "redirect": url_for('index')})
        else:
            # On failure, increment the attempt counter
            session['login_attempts'] = session.get('login_attempts', 0) + 1
            
            if session['login_attempts'] >= 5:
                # If max attempts reached, set a 5-minute lockout
                session['lockout_until'] = time.time() + 300  # 5 minutes
                session.pop('login_attempts', None) # Reset for next cycle
                return jsonify({"status": "error", "msg": "Too many failed attempts. Please wait 5 minutes."}), 429

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