#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversão direta de arquivos .arff (ISCX VPN-nonVPN) para CSV,
ignorando cabeçalhos com erro de @RELATION.
Autor: Leonardo Rodrigues Pereira
"""

import os
import pandas as pd

RAW_PATH = "../data/raw"
OUTPUT_FILE = "ISCX-VPN-nonVPN.csv"

def parse_arff_to_csv(path):
    """Lê um arquivo .arff e retorna um DataFrame pandas, ignorando cabeçalho inválido"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # encontra início dos dados
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("@data"):
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("Arquivo sem seção @DATA")

    # separa as linhas de dados
    data_lines = [l.strip() for l in lines[start_idx:] if l.strip() and not l.startswith('%')]
    data_str = "\n".join(data_lines)

    # tenta detectar delimitador (vírgula ou tab)
    sep = ',' if ',' in data_lines[0] else '\t'

    df = pd.read_csv(pd.io.common.StringIO(data_str), sep=sep, header=None, engine="python")
    return df


def convert_all():
    all_data = []

    for root, _, files in os.walk(RAW_PATH):
        for file in files:
            if file.endswith(".arff"):
                path = os.path.join(root, file)
                print(f"[INFO] Processando {path} ...")

                try:
                    df = parse_arff_to_csv(path)
                    df['Label'] = 'VPN' if 'VPN' in file.upper() else 'nonVPN'
                    all_data.append(df)
                except Exception as e:
                    print(f"[ERRO] Falha ao ler {file}: {e}")

    if not all_data:
        print("[ERRO] Nenhum arquivo válido foi convertido.")
        return

    df_total = pd.concat(all_data, ignore_index=True)
    print(f"[INFO] Dataset combinado: {df_total.shape}")

    output_path = os.path.join(RAW_PATH, OUTPUT_FILE)
    df_total.to_csv(output_path, index=False)
    print(f"[OK] Dataset combinado salvo em: {output_path}")


if __name__ == "__main__":
    convert_all()
