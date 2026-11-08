#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preparar_datos.py — Limpia los CSV de CIC-IDS2017 que usa la Práctica 3.

Entrada : carpeta con los CSV originales (versión MachineLearningCVE).
Salida  : carpeta con un archivo <nombre>_limpio.csv.gz por cada CSV encontrado.

Uso:
    python preparar_datos.py
    python preparar_datos.py --entrada datos_originales --salida datos_limpios
    python preparar_datos.py --entrada "C:\\Mi carpeta\\MachineLearningCVE"

Limpieza (la misma de la práctica 2.2):
  1. Quita los espacios de los nombres de columna y convierte a número lo que sea texto.
  2. Normaliza la etiqueta y crea es_ataque (0 = BENIGN, 1 = cualquier ataque).
  3. Reemplaza infinitos por NaN y elimina esas filas.
  4. Elimina filas duplicadas.
  5. Elimina la columna repetida 'Fwd Header Length.1' y las columnas constantes.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:                                   # evita errores de codificación en la consola de Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                      # noqa: BLE001
    pass

NOMBRES = {
    "web": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "bot": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}
OBLIGATORIOS = ["web", "bot"]
ID_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "SimillarHTTP"]


def limpiar(ruta):
    d = pd.read_csv(ruta, encoding="latin1", low_memory=False)
    d.columns = d.columns.str.strip()
    ids = [c for c in ID_COLS if c in d.columns]          # solo si usaste TrafficLabelling
    d = d.drop(columns=ids)
    for c in d.columns:
        if c != "Label" and not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Label"] = (d["Label"].astype(str).str.strip()
                  .str.replace(r"[^\x20-\x7E]+", "-", regex=True))
    d["es_ataque"] = (d["Label"].str.upper() != "BENIGN").astype(int)
    n0 = len(d)
    num = [c for c in d.columns if c not in ("Label", "es_ataque")]
    d[num] = d[num].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=num).drop_duplicates().reset_index(drop=True)
    n1 = len(d)
    repetidas = [c for c in d.columns if c.endswith(".1") and c[:-2] in d.columns]
    constantes = [c for c in num if c in d.columns and d[c].nunique() <= 1]
    d = d.drop(columns=repetidas + constantes)
    return d, n0, n1, repetidas, constantes, ids


def main():
    ap = argparse.ArgumentParser(description="Limpieza de CIC-IDS2017 para la Práctica 3")
    ap.add_argument("--entrada", nargs="+", default=["datos_originales"],
                    help="carpeta con los CSV originales (MachineLearningCVE)")
    ap.add_argument("--salida", default="datos_limpios", help="carpeta de salida")
    a = ap.parse_args()
    entrada = " ".join(a.entrada)                           # tolera rutas con espacios sin comillas

    if not os.path.isdir(entrada):
        sys.exit(f"ERROR: no existe la carpeta de entrada: {entrada}\n"
                 "Copia allí los CSV (ver LEEME.txt) o indica la ruta con --entrada \"ruta\".")
    os.makedirs(a.salida, exist_ok=True)

    faltan_oblig = []
    for clave, fichero in NOMBRES.items():
        ruta = os.path.join(entrada, fichero)
        print("=" * 78)
        print(f"[{clave}] {fichero}")
        if not os.path.isfile(ruta):
            print("   -> NO ENCONTRADO" + (" (obligatorio)" if clave in OBLIGATORIOS else " (opcional)"))
            if clave in OBLIGATORIOS:
                faltan_oblig.append(fichero)
            continue
        d, n0, n1, rep, cte, ids = limpiar(ruta)
        n_atq = int(d["es_ataque"].sum())
        print(f"   filas: {n0:,} -> {n1:,}  ({n1 / n0 * 100:.2f} % conservado)")
        print(f"   ataques: {n_atq:,} ({n_atq / n1 * 100:.3f} %)   benignos: {n1 - n_atq:,}")
        print(f"   etiquetas: {d['Label'].value_counts().to_dict()}")
        print(f"   columnas eliminadas: repetidas={rep}  constantes={len(cte)}  identificadores={ids}")
        print(f"   características finales: {d.shape[1] - 2}")
        destino = os.path.join(a.salida, f"{clave}_limpio.csv.gz")
        d.to_csv(destino, index=False, compression="gzip")
        print(f"   guardado en: {destino}")

    if faltan_oblig:
        sys.exit("\nFALTAN archivos obligatorios en la carpeta de entrada:\n   " + "\n   ".join(faltan_oblig))
    print("\nListo. Ya puedes ejecutar la práctica:  python practica3_plantilla.py --dataset web")


if __name__ == "__main__":
    main()
