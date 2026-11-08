#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda_cicids2017.py — Práctica 2.2: Análisis exploratorio de CIC-IDS2017

Qué hace (en este orden):
  1. Carga uno o varios CSV del dataset y normaliza los nombres de columnas.
  2. Muestra dimensiones, tipos y columnas duplicadas.
  3. Analiza la distribución de clases (benigno vs. ataque, y por tipo de ataque).
  4. Revisa la calidad del dato: faltantes, infinitos, negativos y columnas constantes.
  5. Limpia (infinitos -> NaN, elimina NaN y duplicados) y compara antes/después.
  6. Compara variables entre clases (estadísticos, histogramas y cajas en escala log).
  7. Calcula correlaciones y lista los pares redundantes.
  8. Mide el poder separador de cada variable (AUC univariado) para detectar atajos.
  9. Analiza los puertos de destino por clase.
 10. Proyecta con PCA para ver si las clases se separan.
 11. Guarda gráficos (PNG), un resumen (resumen_eda.txt) y el dataset limpio.

Carpetas del dataset:
  MachineLearningCVE  -> RECOMENDADA: características + Label (sin IPs ni marcas de tiempo).
  TrafficLabelling    -> también funciona: trae además Flow ID, IPs y Timestamp, que el
                         script muestra (sección 3b) y excluye del análisis.

Uso:
  python eda_cicids2017.py RUTA.csv [RUTA2.csv ...] [--salida eda_salida]
                           [--muestra 20000] [--semilla 42]

  En Jupyter / Colab:  %run eda_cicids2017.py RUTA.csv

Requisitos: pandas, numpy, matplotlib, scikit-learn  (opcional: pyarrow)
"""
import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # guarda figuras sin abrir ventanas
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

COLOR_BEN, COLOR_ATQ = "#2E5597", "#C0392B"
# Columnas que solo trae la carpeta TrafficLabelling: identifican el flujo, no describen el ataque
ID_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "SimillarHTTP"]
VARS_CLAVE = ["Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
              "Flow Bytes/s", "Flow Packets/s", "Packet Length Mean",
              "Flow IAT Mean", "Average Packet Size"]


# ----------------------------------------------------------------- utilidades
class Informe:
    """Imprime en pantalla y acumula el texto para guardarlo en resumen_eda.txt."""
    def __init__(self):
        self.lineas = []

    def __call__(self, texto=""):
        print(texto)
        self.lineas.append(str(texto))

    def titulo(self, texto):
        self("")
        self("=" * 78)
        self(texto)
        self("=" * 78)

    def guardar(self, ruta):
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lineas) + "\n")


def log1p_seguro(x):
    """log10(1 + x) con valores negativos recortados a 0 (escala para colas largas)."""
    return np.log10(1 + np.clip(x, 0, None))


def muestra_balanceada(df, n, semilla):
    """Muestra de hasta n filas con la misma cantidad por clase (si es posible)."""
    por_clase = max(n // 2, 1)
    partes = [g.sample(min(len(g), por_clase), random_state=semilla)
              for _, g in df.groupby("es_ataque")]
    return pd.concat(partes).sample(frac=1, random_state=semilla)


def guardar_fig(fig, carpeta, nombre):
    ruta = os.path.join(carpeta, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    print(f"   [figura] {ruta}")


# ----------------------------------------------------------------- rutas
def resolver_rutas(rutas):
    """Valida las rutas. Si la ruta tenía espacios y no iba entre comillas, intenta recomponerla."""
    if all(os.path.isfile(r) for r in rutas):
        return rutas
    unida = " ".join(rutas)
    if os.path.isfile(unida):
        print(f"Aviso: la ruta tenía espacios y no estaba entre comillas; se usará:\n   {unida}\n")
        return [unida]
    faltan = [r for r in rutas if not os.path.isfile(r)]
    sys.exit("ERROR: no se encontró el archivo:\n   " + "\n   ".join(faltan) + "\n\n"
             "Qué revisar:\n"
             "  1) Si la ruta tiene espacios, enciérrala entre comillas dobles:\n"
             "       python eda_cicids2017.py \"C:\\Mi carpeta\\archivo.csv\"\n"
             "  2) O ubícate en la carpeta del CSV (cd) y usa solo el nombre del archivo.\n"
             "  3) Comprueba que el nombre y la extensión .csv sean exactos.")


# ----------------------------------------------------------------- 1. carga
def cargar(rutas, informe):
    informe.titulo("1. CARGA DE DATOS")
    partes = []
    for r in rutas:
        # encoding latin1: algunas etiquetas traen caracteres no UTF-8
        d = pd.read_csv(r, encoding="latin1", low_memory=False)
        d.columns = d.columns.str.strip()          # los nombres traen espacios
        for c in d.columns:                        # texto como 'Infinity' -> número
            if c not in ID_COLS and c != "Label" and not pd.api.types.is_numeric_dtype(d[c]):
                d[c] = pd.to_numeric(d[c], errors="coerce")
        informe(f"Archivo: {os.path.basename(r)}  ->  {d.shape[0]:,} filas x {d.shape[1]} columnas")
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    if "Label" not in df.columns:
        sys.exit("ERROR: no se encontró la columna 'Label'. ¿Es un CSV de CIC-IDS2017?")
    # Etiqueta limpia: sin espacios ni caracteres raros (p. ej. 'Web Attack – Brute Force')
    df["Label"] = (df["Label"].astype(str).str.strip()
                   .str.replace(r"[^\x20-\x7E]+", "-", regex=True))
    df["es_ataque"] = (df["Label"].str.upper() != "BENIGN").astype(int)
    return df


# ----------------------------------------------------------------- 2. estructura
def estructura(df, informe):
    informe.titulo("2. ESTRUCTURA DEL DATASET")
    informe(f"Filas: {len(df):,}   Columnas: {df.shape[1] - 1} (incluye Label; sin contar es_ataque)")
    informe(f"Memoria aproximada: {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB")
    informe("Tipos de dato:")
    for t, n in df.dtypes.astype(str).value_counts().items():
        informe(f"   {t:10s} {n} columnas")
    # pandas renombra las columnas repetidas como 'Nombre.1'
    duplicadas = [c for c in df.columns if re.search(r"\.\d+$", c)]
    if duplicadas:
        informe(f"Columnas con nombre repetido en el CSV (pandas las renombró): {duplicadas}")
        informe("   -> son copias de otra columna; conviene eliminarlas.")
    else:
        informe("No hay columnas con nombre repetido.")


# ----------------------------------------------------------------- 3. clases
def clases(df, carpeta, informe):
    informe.titulo("3. DISTRIBUCIÓN DE CLASES")
    conteo = df["Label"].value_counts()
    pct = (conteo / len(df) * 100).round(3)
    tabla = pd.DataFrame({"filas": conteo, "porcentaje": pct})
    informe(tabla.to_string())
    n_atq = int(df["es_ataque"].sum())
    n_ben = len(df) - n_atq
    informe("")
    informe(f"Benigno: {n_ben:,} ({n_ben / len(df) * 100:.2f} %)   "
            f"Ataque: {n_atq:,} ({n_atq / len(df) * 100:.2f} %)")
    if n_atq > 0 and n_ben > 0:
        informe(f"Razón clase mayoritaria / minoritaria: {max(n_ben, n_atq) / min(n_ben, n_atq):.1f} a 1")

    fig, ax = plt.subplots(figsize=(8, 4 + 0.15 * len(conteo)))
    colores = [COLOR_BEN if e.upper() == "BENIGN" else COLOR_ATQ for e in conteo.index]
    ax.barh(conteo.index[::-1], conteo.values[::-1], color=colores[::-1])
    ax.set_xscale("log")
    ax.set_xlim(left=1)                    # las barras parten de 1 para no distorsionar
    ax.set_xlabel("Cantidad de flujos (escala logarítmica)")
    ax.set_title("Distribución de clases")
    guardar_fig(fig, carpeta, "01_distribucion_clases.png")


# ----------------------------------------------------------------- 3b. identificadores
def identificadores(df, informe):
    """TrafficLabelling incluye IPs, Flow ID y marcas de tiempo. Se muestran y se excluyen."""
    presentes = [c for c in ID_COLS if c in df.columns]
    if not presentes:
        return df
    informe.titulo("3b. COLUMNAS DE IDENTIFICACIÓN (solo en la carpeta TrafficLabelling)")
    informe(f"Se encontraron: {presentes}")
    informe("Identifican el flujo o el momento del experimento, no el ataque: se EXCLUYEN del análisis.")
    if "Source IP" in df.columns and df["es_ataque"].sum() > 0:
        atq = df.loc[df["es_ataque"] == 1, "Source IP"].value_counts(normalize=True).head(3) * 100
        ben = df.loc[df["es_ataque"] == 0, "Source IP"]
        informe("Origen más frecuente de los flujos de ATAQUE (Source IP):")
        for ip, p_atq in atq.items():
            p_ben = (ben == ip).mean() * 100 if len(ben) else 0.0
            informe(f"   {ip:18s} {p_atq:5.1f} % de los ataques   |   {p_ben:5.1f} % de los benignos")
        informe("Si pocas IPs concentran todos los ataques, un modelo podría 'aprender' la IP del")
        informe("atacante del laboratorio: eso es una FUGA de información, no detección real.")
    return df.drop(columns=presentes)


# ----------------------------------------------------------------- 4. calidad
def calidad(df, num, informe):
    informe.titulo("4. CALIDAD DEL DATO (antes de limpiar)")
    X = df[num]
    nan = X.isna().sum()
    inf = np.isinf(X).sum()
    neg = (X < 0).sum()
    informe(f"Valores faltantes (NaN): {int(nan.sum()):,} en {int((nan > 0).sum())} columnas")
    for c, n in nan[nan > 0].sort_values(ascending=False).head(10).items():
        informe(f"   {c:35s} {n:>10,}")
    informe(f"Valores infinitos (inf): {int(inf.sum()):,} en {int((inf > 0).sum())} columnas")
    for c, n in inf[inf > 0].sort_values(ascending=False).head(10).items():
        informe(f"   {c:35s} {n:>10,}")
    informe(f"Columnas con valores negativos: {int((neg > 0).sum())}")
    for c, n in neg[neg > 0].sort_values(ascending=False).head(10).items():
        informe(f"   {c:35s} {n:>10,} filas negativas")
    if "Flow Duration" in X and "Flow Bytes/s" in X:
        d0 = int((X["Flow Duration"] == 0).sum())
        informe(f"Flujos con 'Flow Duration' = 0: {d0:,}  "
                f"(dividir por duración cero produce inf / NaN en las tasas por segundo)")
    ctes = [c for c in num if X[c].nunique(dropna=True) <= 1]
    informe(f"Columnas constantes (un solo valor): {len(ctes)}")
    for c in ctes:
        informe(f"   {c}")
    return ctes


# ----------------------------------------------------------------- 5. limpieza
def limpiar(df, num, informe):
    informe.titulo("5. LIMPIEZA")
    n0 = len(df)
    antes = df["es_ataque"].value_counts(normalize=True).sort_index() * 100
    por_clase_antes = df["Label"].value_counts()
    df = df.copy()
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=num)
    n1 = len(df)
    columnas_dato = [c for c in df.columns if c != "es_ataque"]
    df = df.drop_duplicates(subset=columnas_dato)
    n2 = len(df)
    informe(f"Filas iniciales:                {n0:>10,}")
    informe(f"Tras quitar inf/NaN:            {n1:>10,}   (se eliminaron {n0 - n1:,})")
    informe(f"Tras quitar duplicados:         {n2:>10,}   (se eliminaron {n1 - n2:,})")
    informe(f"Filas conservadas: {n2 / n0 * 100:.2f} %")
    despues = df["es_ataque"].value_counts(normalize=True).sort_index() * 100
    por_clase_despues = df["Label"].value_counts()
    informe("Filas eliminadas por clase (inf/NaN + duplicados):")
    for lab, n_antes in por_clase_antes.items():
        n_desp = int(por_clase_despues.get(lab, 0))
        informe(f"   {lab:18s} {n_antes:>9,} -> {n_desp:>9,}   ({n_antes - n_desp:,} eliminadas)")
    informe("Proporción de ataques ANTES : %.3f %%" % antes.get(1, 0.0))
    informe("Proporción de ataques DESPUÉS: %.3f %%" % despues.get(1, 0.0))
    return df.reset_index(drop=True)


# ----------------------------------------------------------------- 6. comparación
def comparar(df, carpeta, informe):
    informe.titulo("6. COMPARACIÓN ENTRE CLASES")
    vars_ = [v for v in VARS_CLAVE if v in df.columns][:6]
    resumen = (df.groupby("es_ataque")[vars_].agg(["median", "mean"]).T.round(2))
    resumen.columns = ["benigno", "ataque"]
    informe("Mediana y media por clase (0 = benigno, 1 = ataque):")
    informe(resumen.to_string())

    informe("")
    informe("Variedad: cantidad de valores DISTINTOS por clase (un ataque automatizado suele ser uniforme):")
    variedad = pd.DataFrame({"benigno": df.loc[df["es_ataque"] == 0, vars_].nunique(),
                             "ataque": df.loc[df["es_ataque"] == 1, vars_].nunique()})
    informe(variedad.to_string())

    # Histogramas en escala log10(1+x)
    filas = int(np.ceil(len(vars_) / 3))
    fig, axs = plt.subplots(filas, 3, figsize=(13, 3.6 * filas))
    for ax, v in zip(np.ravel(axs), vars_):
        for e, color, nombre in ((0, COLOR_BEN, "benigno"), (1, COLOR_ATQ, "ataque")):
            x = log1p_seguro(df.loc[df["es_ataque"] == e, v])
            if len(x):
                ax.hist(x, bins=50, alpha=0.55, density=True, color=color, label=nombre)
        ax.set_title(v)
        ax.set_xlabel("log10(1 + valor)")
    np.ravel(axs)[0].legend()
    for ax in np.ravel(axs)[len(vars_):]:
        ax.axis("off")
    guardar_fig(fig, carpeta, "02_histogramas_log.png")

    # Cajas
    fig, axs = plt.subplots(filas, 3, figsize=(13, 3.6 * filas))
    for ax, v in zip(np.ravel(axs), vars_):
        datos = [log1p_seguro(df.loc[df["es_ataque"] == e, v]) for e in (0, 1)]
        try:
            bp = ax.boxplot(datos, tick_labels=["benigno", "ataque"], showfliers=False, patch_artist=True)
        except TypeError:                   # matplotlib < 3.9
            bp = ax.boxplot(datos, labels=["benigno", "ataque"], showfliers=False, patch_artist=True)
        for caja, color in zip(bp["boxes"], (COLOR_BEN, COLOR_ATQ)):
            caja.set_facecolor(color)
            caja.set_alpha(0.55)
        ax.set_title(v)
        ax.set_ylabel("log10(1 + valor)")
    for ax in np.ravel(axs)[len(vars_):]:
        ax.axis("off")
    guardar_fig(fig, carpeta, "03_cajas_log.png")


# ----------------------------------------------------------------- 7. correlación
def correlaciones(muestra, num_var, carpeta, informe, umbral=0.95):
    informe.titulo("7. CORRELACIONES ENTRE VARIABLES")
    corr = muestra[num_var].corr()
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_var)))
    ax.set_xticklabels(num_var, rotation=90, fontsize=5)
    ax.set_yticks(range(len(num_var)))
    ax.set_yticklabels(num_var, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Matriz de correlación (Pearson)")
    guardar_fig(fig, carpeta, "04_correlacion.png")

    sup = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pares = (sup.stack().rename("r").reset_index()
             .rename(columns={"level_0": "var_a", "level_1": "var_b"}))
    altos = pares[pares["r"].abs() >= umbral].sort_values("r", key=np.abs, ascending=False)
    informe(f"Pares de variables con |r| >= {umbral}: {len(altos)} de {len(pares):,} posibles")
    for _, f in altos.head(15).iterrows():
        informe(f"   {f.var_a:32s} {f.var_b:32s} r = {f.r:+.3f}")
    if len(altos) > 15:
        informe(f"   ... y {len(altos) - 15} pares más")
    informe("Interpretación: variables casi idénticas aportan información redundante.")


# ----------------------------------------------------------------- 8. poder separador
def poder_separador(muestra, num_var, carpeta, informe):
    informe.titulo("8. PODER SEPARADOR UNIVARIADO (AUC)")
    y = muestra["es_ataque"].values
    if len(np.unique(y)) < 2:
        informe("Solo hay una clase en la muestra: no se puede calcular el AUC.")
        return
    res = {}
    for c in num_var:
        x = np.nan_to_num(muestra[c].values.astype(float))
        a = roc_auc_score(y, x)
        res[c] = max(a, 1 - a)          # 0.5 = no separa nada, 1.0 = separa solo
    res = pd.Series(res).sort_values(ascending=False)
    informe("AUC de cada variable por sí sola (0,5 = nada; 1,0 = separa perfectamente):")
    for c, a in res.head(15).items():
        marca = "  <-- sospechosa: ¿atajo?" if a >= 0.95 else ""
        informe(f"   {c:35s} {a:.3f}{marca}")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    top = res.head(15)[::-1]
    ax.barh(top.index, top.values, color=COLOR_ATQ)
    ax.axvline(0.5, color="gray", ls="--")
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC univariado")
    ax.set_title("Variables que mejor separan ataque de benigno")
    guardar_fig(fig, carpeta, "05_poder_separador.png")


# ----------------------------------------------------------------- 9. puertos
def puertos(df, carpeta, informe):
    informe.titulo("9. PUERTOS DE DESTINO POR CLASE")
    if "Destination Port" not in df.columns:
        informe("No existe la columna 'Destination Port'.")
        return
    top = df["Destination Port"].value_counts().head(10).index
    tabla = (df[df["Destination Port"].isin(top)]
             .groupby(["Destination Port", "es_ataque"]).size().unstack(fill_value=0))
    tabla = tabla.reindex(columns=[0, 1], fill_value=0)
    tot = df["es_ataque"].value_counts()
    pct = tabla.copy().astype(float)
    pct[0] = tabla[0] / max(tot.get(0, 1), 1) * 100
    pct[1] = tabla[1] / max(tot.get(1, 1), 1) * 100
    pct.columns = ["% de los benignos", "% de los ataques"]
    informe("Los 10 puertos más frecuentes y qué % de cada clase representan:")
    informe(pct.round(2).to_string())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(pct))
    ax.bar(idx - 0.2, pct.iloc[:, 0], 0.4, color=COLOR_BEN, label="benigno")
    ax.bar(idx + 0.2, pct.iloc[:, 1], 0.4, color=COLOR_ATQ, label="ataque")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(p)) for p in pct.index])
    ax.set_xlabel("Puerto de destino")
    ax.set_ylabel("% de la clase")
    ax.legend()
    ax.set_title("Puertos de destino más frecuentes")
    guardar_fig(fig, carpeta, "06_puertos_destino.png")
    atq = df[df["es_ataque"] == 1]
    if len(atq):
        p_top = atq["Destination Port"].value_counts().idxmax()
        pred = (df["Destination Port"] == p_top)
        real = (df["es_ataque"] == 1)
        tp = int((pred & real).sum()); fp = int((pred & ~real).sum())
        fn = int((~pred & real).sum()); tn = int((~pred & ~real).sum())
        informe("")
        informe(f"Regla de UNA sola condición: 'Destination Port == {int(p_top)}' -> ataque")
        informe(f"   TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
        informe(f"   recall={tp / max(tp + fn, 1):.4f}  precisión={tp / max(tp + fp, 1):.4f}  "
                f"exactitud={(tp + tn) / len(df):.4f}")
        informe("Nota: el AUC de la sección 8 no detecta este tipo de atajo, porque el número de")
        informe("puerto no crece ni decrece de forma regular con el ataque (no es monótono).")
    informe("Pregunta clave: ¿el puerto describe el ataque o solo cómo se montó el laboratorio?")


# ----------------------------------------------------------------- 10. PCA
def proyeccion_pca(muestra, num_var, carpeta, informe, semilla):
    informe.titulo("10. PROYECCIÓN PCA (2 componentes)")
    Z = StandardScaler().fit_transform(log1p_seguro(muestra[num_var].values))
    pca = PCA(n_components=2, random_state=semilla).fit(Z)
    P = pca.transform(Z)
    informe(f"Varianza explicada: PC1 = {pca.explained_variance_ratio_[0] * 100:.1f} %, "
            f"PC2 = {pca.explained_variance_ratio_[1] * 100:.1f} %  "
            f"(total {pca.explained_variance_ratio_.sum() * 100:.1f} %)")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for e, color, nombre in ((0, COLOR_BEN, "benigno"), (1, COLOR_ATQ, "ataque")):
        m = muestra["es_ataque"].values == e
        ax.scatter(P[m, 0], P[m, 1], s=6, alpha=0.35, color=color, label=nombre)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA sobre una muestra balanceada (log + estandarización)")
    ax.legend(markerscale=3)
    guardar_fig(fig, carpeta, "07_pca.png")
    informe("Si las clases forman nubes bien separadas, el problema será 'fácil' para casi cualquier modelo.")


# ----------------------------------------------------------------- 11. guardar
def guardar_limpio(df, carpeta, informe):
    informe.titulo("11. SALIDAS")
    base = os.path.join(carpeta, "cic_limpio")
    try:
        df.to_parquet(base + ".parquet", index=False)
        ruta = base + ".parquet"
    except Exception:
        df.to_csv(base + ".csv.gz", index=False, compression="gzip")
        ruta = base + ".csv.gz"
    informe(f"Dataset limpio guardado en: {ruta}")


# ----------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description="EDA de CIC-IDS2017")
    ap.add_argument("rutas", nargs="+", help="uno o más CSV de CIC-IDS2017")
    ap.add_argument("--salida", default="eda_salida", help="carpeta de salida")
    ap.add_argument("--muestra", type=int, default=20000,
                    help="filas de la muestra para correlación, AUC y PCA")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    a.rutas = resolver_rutas(a.rutas)
    os.makedirs(a.salida, exist_ok=True)
    informe = Informe()

    df = cargar(a.rutas, informe)
    estructura(df, informe)
    clases(df, a.salida, informe)
    df = identificadores(df, informe)

    num = [c for c in df.select_dtypes(include=[np.number]).columns if c != "es_ataque"]
    constantes = calidad(df, num, informe)
    df = limpiar(df, num, informe)

    num_var = [c for c in num if c not in constantes and df[c].nunique() > 1]
    informe.titulo("RESUMEN DE VARIABLES PARA MODELAR")
    informe(f"Variables numéricas: {len(num)}   constantes (se omiten en el análisis): "
            f"{len(num) - len(num_var)}   usadas: {len(num_var)}")

    comparar(df, a.salida, informe)
    muestra = muestra_balanceada(df, a.muestra, a.semilla)
    informe(f"\n(Para los pasos 7, 8 y 10 se usa una muestra de {len(muestra):,} filas, "
            f"balanceada por clase)")
    correlaciones(muestra, num_var, a.salida, informe)
    poder_separador(muestra, num_var, a.salida, informe)
    puertos(df, a.salida, informe)
    proyeccion_pca(muestra, num_var, a.salida, informe, a.semilla)
    guardar_limpio(df, a.salida, informe)

    informe.guardar(os.path.join(a.salida, "resumen_eda.txt"))
    print(f"\nListo. Revisa la carpeta '{a.salida}/' (gráficos, resumen y dataset limpio).")


if __name__ == "__main__":
    main()
