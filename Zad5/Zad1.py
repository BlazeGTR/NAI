"""
ZADANIE 1: Klasyfikacja Jakości Wina (Wine Quality Dataset)
Model: Multi-Layer Perceptron (MLP)
Opis: Sieć przewiduje, czy wino jest dobre (ocena >= 6) na podstawie parametrów chemicznych.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def run_wine_task():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Pobieranie danych
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, sep=';')
    
    X = df.drop('quality', axis=1).values
    y = (df['quality'] >= 6).astype(int).values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tensory
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    # Model
    model = nn.Sequential(
        nn.Linear(11, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    print("Trening modelu...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (model(X_test_t) > 0.5).float().cpu()
        print("\n--- RAPORT: WINE QUALITY (MLP) ---")
        print(classification_report(y_test, preds))

if __name__ == "__main__":
    run_wine_task()