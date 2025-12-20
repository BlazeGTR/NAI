"""
ZADANIE 1 (Rozszerzone): Porównanie architektur i Macierz Pomyłek.
Ten skrypt trenuje dwie sieci: 'Małą' oraz 'Dużą' i generuje Confusion Matrix.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 1. Przygotowanie danych
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(url, sep=';')
X = df.drop('quality', axis=1).values
y = (df['quality'] >= 6).astype(int).values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_test_t = torch.FloatTensor(X_test)

# 2. Definicja dwóch różnych rozmiarów sieci
small_model = nn.Sequential(
    nn.Linear(11, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

large_model = nn.Sequential(
    nn.Linear(11, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

def train_and_eval(model, name):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    print(f"\nTrenowanie modelu: {name}...")
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        preds = (model(X_test_t) > 0.5).float().numpy()
        return preds

# 3. Uruchomienie porównania
preds_small = train_and_eval(small_model, "MAŁA SIEĆ (1 warstwa ukryta, 8 neuronów)")
preds_large = train_and_eval(large_model, "DUŻA SIEĆ (3 warstwy ukryte, do 128 neuronów)")

# 4. Logi / Wyniki
print("\n--- PORÓWNANIE WYNIKÓW ---")
print("MAŁA SIEĆ:")
print(classification_report(y_test, preds_small))
print("\nDUŻA SIEĆ:")
print(classification_report(y_test, preds_large))

# 5. Confusion Matrix dla Dużej Sieci
cm = confusion_matrix(y_test, preds_large)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Słabe', 'Dobre'], yticklabels=['Słabe', 'Dobre'])
plt.xlabel('Przewidziane')
plt.ylabel('Rzeczywiste')
plt.title('Confusion Matrix - Duża Sieć (Wine Quality)')
plt.show()