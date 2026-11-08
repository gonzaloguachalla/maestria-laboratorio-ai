"""Práctica 1 — Regla vs modelo"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

df = pd.read_csv("actividad_usuarios.csv")
print(df.head(), "\n")
print(df["is_attack"].value_counts().rename({0: "normal", 1: "ataque"}), "\n")

X = df[["failed_logins", "hour", "mb_out"]]   # lo que ve el modelo
y = df["is_attack"]                           # solo para evaluar

# 1) Regla clásica de fuerza bruta: más de 5 intentos fallidos
regla = (X["failed_logins"] > 5).astype(int)
print("Regla (filas = real, columnas = predicho):")
print(confusion_matrix(y, regla), "\n")

# 2) Modelo no supervisado (no usa las etiquetas para aprender)
iso = IsolationForest(contamination=0.02, random_state=42).fit(X)
modelo = (iso.predict(X) == -1).astype(int)
print("Modelo (Isolation Forest):")
print(confusion_matrix(y, modelo))
