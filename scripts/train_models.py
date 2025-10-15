#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treinamento e avaliação de modelos de machine learning
para detecção de tráfego WebRTC em ambientes criptografados.

Autor: Leonardo Rodrigues Pereira
UTFPR - Mestrado em Telecomunicações e Inteligência Artificial
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, silhouette_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# ============================================================
# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/dataset_clean.csv")
FIGURES_PATH = os.path.join(BASE_DIR, "reports/figures")
os.makedirs(FIGURES_PATH, exist_ok=True)
# ============================================================

def load_data():
    print(f"[INFO] Carregando dataset processado de {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dimensões: {df.shape}")

    # Identifica a coluna de rótulo
    label_col = "Label" if "Label" in df.columns else df.columns[-1]

    X = df.drop(columns=[label_col])
    y = df[label_col].apply(lambda x: 1 if str(x).lower() == "vpn" else 0)
    return X, y


def train_random_forest(X_train, y_train, X_test, y_test):
    print("[INFO] Treinando Random Forest...")
    rf = RandomForestClassifier(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.3f}")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest - Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "rf_confusion_matrix.png"))
    plt.close()

    # Importância das features
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    top_features.plot(kind='barh', title='Top 10 Features - Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "rf_feature_importance.png"))
    plt.close()

    return auc


def train_svm(X_train, y_train, X_test, y_test):
    print("[INFO] Treinando SVM...")
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title("SVM - Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "svm_confusion_matrix.png"))
    plt.close()

    return auc


def train_kmeans(X, y):
    print("[INFO] Treinando K-Means (não supervisionado)...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)

    score = silhouette_score(X, clusters)
    corr = np.corrcoef(clusters, y)[0, 1]
    print(f"Silhouette Score: {score:.3f}")
    print(f"Correlação com rótulo real: {corr:.3f}")

    return score


def train_isolation_forest(X, y):
    print("[INFO] Treinando Isolation Forest (detecção de anomalias)...")
    iso = IsolationForest(contamination=0.1, random_state=42)
    preds = iso.fit_predict(X)
    preds = np.where(preds == -1, 1, 0)
    corr = np.corrcoef(preds, y)[0, 1]
    print(f"Correlação com rótulo real: {corr:.3f}")
    return corr


def main():
    X, y = load_data()

    # Divide treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Normalização adicional (garantia)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Modelos supervisionados
    auc_rf = train_random_forest(pd.DataFrame(X_train, columns=X.columns), y_train,
                                 pd.DataFrame(X_test, columns=X.columns), y_test)
    auc_svm = train_svm(pd.DataFrame(X_train, columns=X.columns), y_train,
                        pd.DataFrame(X_test, columns=X.columns), y_test)

    # Modelos não supervisionados
    score_km = train_kmeans(X, y)
    corr_iso = train_isolation_forest(X, y)

    print("\n=== RESUMO FINAL ===")
    print(f"Random Forest AUC: {auc_rf:.3f}")
    print(f"SVM AUC: {auc_svm:.3f}")
    print(f"K-Means Silhouette: {score_km:.3f}")
    print(f"Isolation Forest Corr: {corr_iso:.3f}")
    print("\n[OK] Resultados e figuras salvos em 'reports/figures/'")


if __name__ == "__main__":
    main()


