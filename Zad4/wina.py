"""
Autor:
Błażej Majchrzak

Predykcja jakościn wina
Decision Tree + SVM + Kernel + PCA

Wymagania:

!pip install pandas scikit-learn matplotlib joblib

ten skrypt robi:
1. Pobranie danych Wine Quality (red) z UCI.
2. Utworzenie etykiety binarnej: quality >= 6 => 1 (good), inaczej 0 (bad).
3. Podział danych.
4. Standaryzację.
5. Trening Decision Tree.
6. Trening podstawowego SVM (RBF).
7. GridSearchCV dla wielu kernelów SVM.
8. Wizualizację PCA.
9. Przykładową predykcję.

Dane:
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA


def load_data():
    """
    Wczytuje dane Wine Quality (red) i tworzy etykietę binarną.

    Returns:
        DataFrame: oryginalne dane z dodatkową kolumną 'target'
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    df["target"] = (df["quality"] >= 6).astype(int)
    print("Sample of data:")
    print(df.head())
    return df


def split_data(df):
    """
    Dzieli dane na zbiory treningowy i testowy.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["quality", "target"])
    y = df["target"]

    return train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )


def scale_features(X_train, X_test):
    """
    Standaryzuje dane wejściowe.

    Returns:
        tuple: X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def train_decision_tree(X_train, y_train):
    """
    Trenuje klasyfikator Decision Tree.

    Returns:
        trained model
    """
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    return dt


def train_basic_svm(X_train_scaled, y_train):
    """
    Trenuje podstawowy model SVM (RBF).

    Returns:
        trained model
    """
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_train_scaled, y_train)
    return svm


def run_grid_search(X_train_scaled, y_train):
    """
    Przeprowadza GridSearchCV dla różnych kernelów SVM.

    Returns:
        tuple: best_model, best_params, grid_object
    """
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1]},
        {'kernel': ['poly'], 'C': [0.1, 1], 'degree': [2, 3], 'gamma': ['scale', 0.1]},
        {'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['scale', 0.1]},
    ]

    grid = GridSearchCV(SVC(), param_grid, cv=4, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    return grid.best_estimator_, grid.best_params_, grid


def visualize_pca(X, y):
    """
    Wizualizuje dane w przestrzeni PCA 2D.

    Args:
        X (DataFrame): cechy
        y (Series): etykiety
    """
    pca = PCA(n_components=2)
    X_red = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_red[:, 0], X_red[:, 1], c=y, alpha=0.6, cmap="coolwarm")
    plt.title("PCA 2D: Wine Quality (binary)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


def example_prediction(df, scaler, dt, svm, best_svm):
    """
    Predykcja dla pierwszej próbki danych.

    Args:
        df (DataFrame): oryginalny zbiór danych
        scaler: StandardScaler
        dt: Decision Tree model
        svm: podstawowy SVM
        best_svm: najlepszy model z GridSearch
    """
    X = df.drop(columns=["quality", "target"])
    sample = X.iloc[[0]]
    sample_scaled = scaler.transform(sample)

    print("\nExample wine sample prediction:")
    print("Decision Tree:", dt.predict(sample)[0])
    print("SVM (RBF):", svm.predict(sample_scaled)[0])
    print("Best SVM:", best_svm.predict(sample_scaled)[0])


# ------------------------------------------
# MAIN 
# ------------------------------------------

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

dt = train_decision_tree(X_train, y_train)
svm = train_basic_svm(X_train_scaled, y_train)

dt_pred = dt.predict(X_test)
svm_pred = svm.predict(X_test_scaled)

print("\nDecision Tree accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

print("\nSVM (RBF) accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

best_svm, best_params, grid_obj = run_grid_search(X_train_scaled, y_train)
best_pred = best_svm.predict(X_test_scaled)

print("\nBest SVM accuracy:", accuracy_score(y_test, best_pred))
print("Best params:", best_params)
print(classification_report(y_test, best_pred))

visualize_pca(df.drop(columns=["quality", "target"]), df["target"])

example_prediction(df, scaler, dt, svm, best_svm)
