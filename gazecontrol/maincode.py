import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

# Webcam
cam = cv2.VideoCapture(0)

# FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
)

screen_w, screen_h = pyautogui.size()

# Cursor position
cursor_x, cursor_y = screen_w // 2, screen_h // 2
SMOOTH = 0.35   # lower = smoother | higher = faster cursor

# Blink detection variables
BLINK_THRESHOLD = 5       # pixels
COOLDOWN = 0.9            # seconds
last_click = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:

        lm = result.multi_face_landmarks[0].landmark

        # --------------------------------------
        # 1️⃣ EYE BLINK DETECTION (PIXEL METHOD)
        # --------------------------------------
        # left eye top/bottom
        L_top = int(lm[159].y * h)
        L_bottom = int(lm[145].y * h)

        # right eye top/bottom
        R_top = int(lm[386].y * h)
        R_bottom = int(lm[374].y * h)

        left_open = abs(L_top - L_bottom)
        right_open = abs(R_top - R_bottom)

        # Debug text
        cv2.putText(frame, f"L:{left_open}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"R:{right_open}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        now = time.time()

        if left_open < BLINK_THRESHOLD and right_open < BLINK_THRESHOLD:
            if now - last_click > COOLDOWN:
                print("✔ Blink Click")
                pyautogui.click()
                last_click = now

        # --------------------------------------
        # 2️⃣ FULL-SCREEN MOUSE MOVEMENT (FIXED)
        # --------------------------------------

        # Use iris center for accurate tracking
        ix = int(lm[473].x * w)
        iy = int(lm[473].y * h)

        # Draw point
        cv2.circle(frame, (ix, iy), 4, (0, 255, 0), -1)

        # Convert to full monitor screen coordinates
        target_x = screen_w * (ix / w)
        target_y = screen_h * (iy / h)

        # Smooth movement
        cursor_x += (target_x - cursor_x) * SMOOTH
        cursor_y += (target_y - cursor_y) * SMOOTH

        # Move cursor
        pyautogui.moveTo(cursor_x, cursor_y)

    cv2.imshow("Eye Mouse Control", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit.")
        break

cam.release()
cv2.destroyAllWindows()
