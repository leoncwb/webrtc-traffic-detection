#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pré-processamento do dataset ISCX VPN-nonVPN / WebRTC
Autor: Leonardo Rodrigues Pereira
UTFPR - Mestrado em Telecomunicações e IA

Este script lê um CSV de fluxos de rede, remove colunas irrelevantes,
trata valores ausentes, normaliza as variáveis numéricas e salva a
versão processada em data/processed/.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Caminhos
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
INPUT_FILE = "ISCX-VPN-nonVPN.csv"         # nome do arquivo original
OUTPUT_FILE = "dataset_clean.csv"          # nome do arquivo processado
# ---------------------------------------------------------

def load_dataset():
    """Carrega o dataset bruto (CSV)"""
    csv_path = os.path.join(RAW_PATH, INPUT_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    print(f"[INFO] Carregando dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dimensões iniciais: {df.shape}")
    return df


def clean_dataset(df):
    """Remove colunas vazias e trata valores ausentes"""
    print("[INFO] Limpando dataset...")

    # Remove colunas com mais de 40% de valores ausentes
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.4].index
    df = df.drop(columns=cols_to_drop)

    # Substitui valores ausentes restantes pela mediana da coluna
    df = df.fillna(df.median(numeric_only=True))

    # Remove colunas não numéricas irrelevantes (ex: IPs, portas, timestamps)
    drop_candidates = [col for col in df.columns if df[col].dtype == "object" and col.lower() != "label"]
    df = df.drop(columns=drop_candidates)

    print(f"[INFO] Dimensões após limpeza: {df.shape}")
    return df


def normalize_features(df):
    """Normaliza colunas numéricas (z-score)"""
    print("[INFO] Normalizando variáveis numéricas...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def save_dataset(df):
    """Salva o dataset processado"""
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_PATH, OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Dataset processado salvo em: {output_path}")


def main():
    df = load_dataset()
    df = clean_dataset(df)
    df = normalize_features(df)
    save_dataset(df)


if __name__ == "__main__":
    main()


