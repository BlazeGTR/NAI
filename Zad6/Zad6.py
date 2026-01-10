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
    Nakłada obraz PNG (overlay) na obraz tła (background) z uwzględnieniem:
    - skalowania,
    - rotacji,
    - kanału alfa (przezroczystości).

    Parametry
    ----------
    background : np.ndarray
        Obraz tła (BGR), na który nakładany jest overlay.
    overlay : np.ndarray
        Obraz PNG z kanałem alfa (BGRA).
    x, y : int
        Współrzędne lewego górnego rogu nakładki na obrazie tła.
    w, h : int
        Szerokość i wysokość nakładki po przeskalowaniu.
    angle : float
        Kąt rotacji w stopniach (zgodnie z ruchem wskazówek zegara).

    Zwraca
    -------
    np.ndarray
        Obraz tła z nałożonym overlayem.
    """
    if overlay is None:
        return background

    # 1. Skalowanie nakładki
    overlay_res = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    # 2. Rotacja nakładki wokół środka
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    overlay_rot = cv2.warpAffine(
        overlay_res,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # 3. Rozdzielenie kanałów (BGR + alfa)
    b, g, r, a = cv2.split(overlay_rot)
    overlay_color = cv2.merge((b, g, r))
    mask = a.astype(float) / 255.0

    # 4. Wyznaczenie obszaru roboczego (zabezpieczenie przed wyjściem poza kadr)
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)

    overlay_x1, overlay_x2 = max(0, -x), min(w, background.shape[1] - x)
    overlay_y1, overlay_y2 = max(0, -y), min(h, background.shape[0] - y)

    if x1 >= x2 or y1 >= y2:
        return background

    # 5. Mieszanie kolorów z użyciem maski alfa
    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            mask[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
            * overlay_color[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c]
            + (1.0 - mask[overlay_y1:overlay_y2, overlay_x1:overlay_x2])
            * background[y1:y2, x1:x2, c]
        )

    return background


# =======================
# KONFIGURACJA DETEKCJI
# =======================

"""
Ładowanie klasyfikatorów Haar Cascade do:
- detekcji twarzy,
- detekcji oczu.
"""
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# =======================
# ŁADOWANIE ZASOBÓW
# =======================

"""
Obrazy PNG muszą zawierać kanał alfa (przezroczystość).
"""
img_glasses = cv2.imread('okulary.png', cv2.IMREAD_UNCHANGED)
img_earring = cv2.imread('kolczyk.png', cv2.IMREAD_UNCHANGED)

video_path = 'film.mp4'
cap = cv2.VideoCapture(video_path)

TARGET_WIDTH = 480  # Docelowa szerokość klatki wideo


# =======================
# GŁÓWNA PĘTLA WIDEO
# =======================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skalowanie klatki do stałej szerokości
    h_orig, w_orig = frame.shape[:2]
    ratio = TARGET_WIDTH / float(w_orig)
    frame = cv2.resize(
        frame,
        (TARGET_WIDTH, int(h_orig * ratio)),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcja twarzy
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

        current_angle = 0.0

        # =======================
        # OKULARY
        # =======================
        if len(eyes) >= 2:
            # Sortowanie oczu od lewego do prawego
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[-1]

            # Środki oczu
            p1 = (ex1 + ew1 // 2, ey1 + eh1 // 2)
            p2 = (ex2 + ew2 // 2, ey2 + eh2 // 2)

            # Obliczanie kąta nachylenia głowy
            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            current_angle = math.degrees(math.atan2(dy, dx))

            # Pozycjonowanie i rozmiar okularów
            g_w = int(w * 0.8)
            g_h = int(g_w * 0.4)
            g_x = x + int(w * 0.1)
            g_y = y + int((ey1 + ey2) / 2)

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
        """
        Zakładamy, że:
        - płatki uszu znajdują się na ~65% wysokości twarzy,
        - kolczyki są lekko poza obrysem twarzy.
        """
        earring_size = int(w * 0.1)
        earring_size = max(earring_size, 1)

        # Lewe ucho (z perspektywy kamery)
        ear_l_x = x - int(earring_size * 0.2)
        ear_l_y = y + int(h * 0.65)

        # Prawe ucho (z perspektywy kamery)
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
