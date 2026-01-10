"""
Wirtualna Przymierzalnia Akcesoriów (Augmented Reality)
Autor: Błażej Majchrzak

Opis:
    Program implementuje system nakładania wirtualnych akcesoriów (okulary, kolczyki) 
    na twarz w czasie rzeczywistym lub w pliku wideo. Wykorzystuje bibliotekę OpenCV 
    oraz klasyfikatory Haar Cascade do detekcji kluczowych punktów twarzy.
    
    Główne funkcjonalności:
        - Automatyczna detekcja twarzy i oczu w strumieniu wideo.
        - Dynamiczne skalowanie i rotacja nakładek PNG (z kanałem alfa) 
          na podstawie kąta nachylenia linii oczu.
        - Zaawansowane nakładanie obrazów (alpha blending) zapewniające 
          płynne przejścia i przezroczystość.
        - Obsługa akcesoriów bocznych (kolczyki) pozycjonowanych relatywnie 
          do wymiarów wykrytej twarzy.

Potrzebne biblioteki:
    pip install opencv-python numpy

Instrukcja użycia:
    1. Prześlij pliki „okulary.png”, „kolczyk.png” oraz „film.mp4” do środowiska (np. Google Colab).
    2. Uruchom skrypt.
    3. Program przetworzy wideo klatka po klatce, nakładając dodatki na każdą wykrytą twarz.
    4. Wynik:
        – podgląd wideo z naniesionymi akcesoriami wyświetlany w czasie rzeczywistym,
        – automatyczne dopasowanie kąta okularów do nachylenia głowy użytkownika.
"""

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from IPython.display import clear_output
import math


def overlay_png(background, overlay, x, y, w, h, angle):
    """
    Nakłada obraz PNG (overlay) na obraz tła (background) z obsługą:
    - skalowania,
    - rotacji,
    - przezroczystości (kanał alfa).
    """
    if overlay is None:
        return background

    overlay_res = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    overlay_rot = cv2.warpAffine(
        overlay_res,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    b, g, r, a = cv2.split(overlay_rot)
    overlay_rgb = cv2.merge((b, g, r))
    mask = a.astype(float) / 255.0

    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)

    ox1, ox2 = max(0, -x), min(w, background.shape[1] - x)
    oy1, oy2 = max(0, -y), min(h, background.shape[0] - y)

    if x1 >= x2 or y1 >= y2:
        return background

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            mask[oy1:oy2, ox1:ox2] * overlay_rgb[oy1:oy2, ox1:ox2, c]
            + (1.0 - mask[oy1:oy2, ox1:ox2]) * background[y1:y2, x1:x2, c]
        )

    return background


# =======================
# KONFIGURACJA
# =======================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

img_glasses = cv2.imread('okulary.png', cv2.IMREAD_UNCHANGED)
img_earring = cv2.imread('kolczyk.png', cv2.IMREAD_UNCHANGED)

video_path = 'film.mp4'
cap = cv2.VideoCapture(video_path)

TARGET_WIDTH = 480

# Zapamiętane parametry okularów (fallback)
last_glasses = {
    "rel_x": None,
    "rel_y": None,
    "w": None,
    "h": None,
    "angle": 0.0
}


# =======================
# GŁÓWNA PĘTLA
# =======================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h_orig, w_orig = frame.shape[:2]
    ratio = TARGET_WIDTH / float(w_orig)
    frame = cv2.resize(
        frame,
        (TARGET_WIDTH, int(h_orig * ratio)),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

        g_w = g_h = g_x = g_y = None
        current_angle = last_glasses["angle"]

        # =======================
        # OKULARY – TRYB HYBRYDOWY
        # =======================

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[-1]

            p1 = (ex1 + ew1 // 2, ey1 + eh1 // 2)
            p2 = (ex2 + ew2 // 2, ey2 + eh2 // 2)

            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            current_angle = math.degrees(math.atan2(dy, dx))

            g_w = int(w * 0.8)
            g_h = int(g_w * 0.4)
            g_x = x + int(w * 0.1)
            g_y = y + int((ey1 + ey2) / 2)

            # zapamiętaj proporcje względem twarzy
            last_glasses["rel_x"] = (g_x - x) / w
            last_glasses["rel_y"] = (g_y - y) / h
            last_glasses["w"] = g_w / w
            last_glasses["h"] = g_h / h
            last_glasses["angle"] = current_angle

        # fallback – brak oczu
        elif last_glasses["rel_x"] is not None:
            g_w = int(w * last_glasses["w"])
            g_h = int(h * last_glasses["h"])
            g_x = x + int(w * last_glasses["rel_x"])
            g_y = y + int(h * last_glasses["rel_y"])
            current_angle = last_glasses["angle"]

        if g_w is not None:
            frame = overlay_png(
                frame,
                img_glasses,
                g_x,
                g_y,
                g_w,
                g_h,
                -current_angle
            )

        # =======================
        # KOLCZYKI
        # =======================

        earring_size = max(int(w * 0.1), 1)

        ear_l_x = x - int(earring_size * 0.2)
        ear_l_y = y + int(h * 0.65)

        ear_r_x = x + w - int(earring_size * 0.8)
        ear_r_y = y + int(h * 0.65)

        frame = overlay_png(
            frame,
            img_earring,
            ear_l_x,
            ear_l_y,
            earring_size,
            earring_size,
            -current_angle
        )
        frame = overlay_png(
            frame,
            img_earring,
            ear_r_x,
            ear_r_y,
            earring_size,
            earring_size,
            -current_angle
        )

    clear_output(wait=True)
    cv2_imshow(frame)

cap.release()
