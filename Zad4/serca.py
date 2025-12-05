"""
Autor:
Błażej Majchrzak

Predykcja chorób serca
Decision Tree + SVM + Kernel + PCA

Wymagania:

!pip install pandas scikit-learn matplotlib joblib

ten skrypt robi:
1. Pobiera dane dotyczące chorób serca.
2. Dzieli dane na zbiory treningowy i testowy.
3. Standaryzuje cechy (wymagane dla SVM).
4. Trenuje model drzewa decyzyjnego.
5. Trenuje podstawowy model SVM (RBF).
6. Przeprowadza GridSearchCV na wielu kernelach SVM.
7. Tworzy wizualizację PCA.
8. Przeprowadza przykładową predykcję dla jednego pacjenta.

Źródło danych:
https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA


def load_data():
    """
    Wczytuje zbiór danych dotyczący chorób serca.

    Returns:
        DataFrame: kompletna tabela danych.
    """
    url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
    df = pd.read_csv(url)
    print("Sample of data:")
    print(df.head())
    return df


def split_data(df):
    """
    Dzieli dane na zbiory treningowy i testowy.

    Args:
        df (DataFrame): dane wejściowe.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def scale_features(X_train, X_test):
    """
    Standaryzuje dane, co jest wymagane w SVM.

    Args:
        X_train (ndarray): cechy treningowe
        X_test (ndarray): cechy testowe

    Returns:
        tuple: X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_decision_tree(X_train, y_train):
    """
    Trenuje klasyfikator Decision Tree.

    Returns:
        model: wytrenowany klasyfikator
    """
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    return tree


def train_basic_svm(X_train_scaled, y_train):
    """
    Trenuje podstawowy model SVM z jądrem RBF.

    Returns:
        model: wytrenowany klasyfikator
    """
    svm = SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    return svm


def run_grid_search(X_train_scaled, y_train):
    """
    Przeprowadza GridSearchCV dla wielu kernelów SVM.

    Returns:
        tuple: najlepszy model, parametry, obiekt GridSearchCV
    """
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.1, 1, 10, 50],
        'gamma': ['scale', 0.1, 0.01]
    }

    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    return best_model, grid.best_params_, grid


def visualize_pca(X, y):
    """
    Generuje dwuwymiarową projekcję PCA i wykres.

    Args:
        X (DataFrame): cechy
        y (Series): etykiety
    """
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='coolwarm', s=25)
    plt.title("PCA Visualization of Heart Disease Data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


def example_prediction(X, scaler, tree, svm, best_svm):
    """
    Wykonuje predykcję dla przykładowego pacjenta (pierwszy wiersz danych).

    Args:
        X (DataFrame): cechy
        scaler: obiekt StandardScaler
        tree: model decyzjny
        svm: podstawowy model SVM
        best_svm: najlepszy model z GridSearch
    """
    sample = X.iloc[[0]]
    sample_scaled = scaler.transform(sample)

    print("\nExample patient prediction:")
    print("Decision Tree:", tree.predict(sample)[0])
    print("SVM (RBF):", svm.predict(sample_scaled)[0])
    print("Best SVM:", best_svm.predict(sample_scaled)[0])


# ─────────────────────────────────────────────────────────
# MAIN SCRIPT
# ─────────────────────────────────────────────────────────

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

tree = train_decision_tree(X_train, y_train)
svm = train_basic_svm(X_train_scaled, y_train)

tree_pred = tree.predict(X_test)
svm_pred = svm.predict(X_test_scaled)

print("\nDecision Tree accuracy:", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))

print("\nSVM (RBF) accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

best_svm, best_params, grid = run_grid_search(X_train_scaled, y_train)
best_pred = best_svm.predict(X_test_scaled)

print("\nBest SVM accuracy:", accuracy_score(y_test, best_pred))
print("Best parameters:", best_params)
print(classification_report(y_test, best_pred))

visualize_pca(df.drop("target", axis=1), df["target"])

example_prediction(df.drop("target", axis=1), scaler, tree, svm, best_svm)
