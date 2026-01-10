# Wirtualna Przymierzalnia Akcesoriów

Aplikacja do nakładania wirtualnych akcesoriów (okulary, kolczyki) na twarz w czasie rzeczywistym przy użyciu OpenCV i detekcji twarzy Haar Cascade.

## Funkcje

- Automatyczna detekcja twarzy i oczu
- Dynamiczne dopasowanie kąta okularów do nachylenia głowy
- Obsługa przezroczystych nakładek PNG
- Pozycjonowanie kolczyków względem twarzy
- Przetwarzanie wideo klatka po klatce

## Wymagania

```bash
pip install opencv-python numpy
```

## Użycie

1. Przygotuj pliki:
   - okulary.png - obraz okularów z przezroczystym tłem
   - kolczyk.png - obraz kolczyka z przezroczystym tłem
   - film.mp4 - wideo do przetworzenia

2. Uruchom skrypt:
```python
python virtual_fitting.py
```

3. Program automatycznie nałoży akcesoria na każdą wykrytą twarz w wideo

## Jak to działa

- Detekcja twarzy: Haar Cascade wykrywa twarze w każdej klatce
- Detekcja oczu: Lokalizuje oczy i oblicza kąt nachylenia głowy
- Fallback: Jeśli oczy nie są widoczne, używa ostatnich znanych pozycji
- Alpha blending: Nakłada obrazy z zachowaniem przezroczystości

## Autor

Błażej Majchrzak

## Notatki

Skrypt został zaprojektowany do użycia w Google Colab, ale działa również lokalnie po drobnych modyfikacjach (usunięcie cv2_imshow i clear_output).
