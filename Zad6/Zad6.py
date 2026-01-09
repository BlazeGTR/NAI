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
    Nakłada obraz PNG z kanałem alfa na tło, uwzględniając rotację, skalowanie i przezroczystość.

    Args:
        background (numpy.ndarray): Obraz tła (klatka wideo), na który nakładamy element.
        overlay (numpy.ndarray): Obraz PNG do nałożenia (musi posiadać 4 kanały: BGRA).
        x (int): Współrzędna X lewego górnego rogu, gdzie ma pojawić się nakładka.
        y (int): Współrzędna Y lewego górnego rogu.
        w (int): Docelowa szerokość nakładki po skalowaniu.
        h (int): Docelowa wysokość nakładki po skalowaniu.
        angle (float): Kąt obrotu nakładki w stopniach (zgodnie z ruchem wskazówek zegara).

    Returns:
        numpy.ndarray: Obraz z naniesioną nakładką.
    """
    if overlay is None:
        return background
    
    # 1. Skalowanie i przygotowanie macierzy rotacji
    overlay_res = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 2. Wykonanie rotacji z zachowaniem przezroczystości tła (borderValue=(0,0,0,0))
    overlay_rot = cv2.warpAffine(
        overlay_res, rotation_matrix, (w, h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0,0,0,0)
    )
    
    # 3. Separacja kanałów i tworzenie maski alfa
    b, g, r, a = cv2.split(overlay_rot)
    overlay_color = cv2.merge((b, g, r))
    mask = a.astype(float) / 255.0

    # 4. Obliczanie granic nakładania (zabezpieczenie przed wyjściem poza krawędzie obrazu)
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    
    overlay_x1, overlay_x2 = max(0, -x), min(w, background.shape[1] - x)
    overlay_y1, overlay_y2 = max(0, -y), min(h, background.shape[0] - y)

    if x1 >= x2 or y1 >= y2:
        return background

    # 5. Mieszanie (Alpha Blending) - nakładanie warstwowe
    for c in range(0, 3):
        bg_slice = background[y1:y2, x1:x2, c]
        ov_slice = overlay_color[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c]
        alpha = mask[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        
        background[y1:y2, x1:x2, c] = (alpha * ov_slice + (1.0 - alpha) * bg_slice)
        
    return background

# --- INICJALIZACJA DETEKTORÓW ---
# Załadowanie gotowych modeli Haar Cascade do wykrywania twarzy i oczu
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --- WCZYTYWANIE ZASOBÓW ---
# IMREAD_UNCHANGED jest kluczowe, aby wczytać 4. kanał (alfa) obrazów PNG
img_glasses = cv2.imread('okulary.png', cv2.IMREAD_UNCHANGED)
img_earring = cv2.imread('kolczyk.png', cv2.IMREAD_UNCHANGED)

# Konfiguracja źródła wideo
video_path = 'film.mp4'
cap = cv2.VideoCapture(video_path)
TARGET_WIDTH = 480 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skalowanie klatki dla zachowania płynności przetwarzania
    h_orig, w_orig = frame.shape[:2]
    ratio = TARGET_WIDTH / float(w_orig)
    frame = cv2.resize(frame, (TARGET_WIDTH, int(h_orig * ratio)), interpolation=cv2.INTER_AREA)
    
    # Konwersja do odcieni szarości (wymagana przez detektory Haar)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        """Przetwarzanie każdej wykrytej twarzy w klatce."""
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        
        current_angle = 0
        
        # --- LOGIKA DLA OKULARÓW ---
        if len(eyes) >= 2:
            # Sortowanie oczu od lewej do prawej
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[-1]
            
            # Obliczanie punktów centralnych oczu
            p1 = (ex1 + ew1//2, ey1 + eh1//2)
            p2 = (ex2 + ew2//2, ey2 + eh2//2)