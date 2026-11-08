# -*- coding: utf-8 -*-
"""
p3_utils.py — Utilidades de la Práctica 3 (NO necesitas modificar este archivo).

Contiene: carga de datos limpios, partición entrenamiento/validación/prueba,
el modelo SVM, simulación de prevalencia y TODAS las funciones de gráficos.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # los gráficos se guardan como PNG
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import plot_tree

# Archivos del dataset que se usan en la práctica (carpeta MachineLearningCVE)
NOMBRES = {
    "web": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",   # principal
    "bot": "Friday-WorkingHours-Morning.pcap_ISCX.csv",                # extensión
    "ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",        # opcional (contraste)
}
TITULOS = {"web": "Ataques web (jueves por la mañana)",
           "bot": "Botnet (viernes por la mañana)",
           "ddos": "DDoS (viernes por la tarde)"}
AZUL, ROJO, VERDE, NARANJA, GRIS = "#2E5597", "#C0392B", "#2E7D32", "#E67E22", "#7F7F7F"


# ------------------------------------------------------------------ datos
def cargar_limpio(nombre, carpeta="datos_limpios"):
    """Carga <nombre>_limpio.csv.gz generado por preparar_datos.py."""
    ruta = os.path.join(carpeta, f"{nombre}_limpio.csv.gz")
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}.\nEjecuta primero:  python preparar_datos.py")
    return pd.read_csv(ruta)


def separar_xy(df):
    """Devuelve X (solo características numéricas) e y (0 = benigno, 1 = ataque)."""
    y = df["es_ataque"].to_numpy()
    X = df.drop(columns=["Label", "es_ataque"])
    return X, y


def particion(y, semilla=42):
    """Partición ESTRATIFICADA 60 % entrenamiento / 20 % validación / 20 % prueba.
    Devuelve tres arreglos de índices (idx_tr, idx_va, idx_te)."""
    idx = np.arange(len(y))
    idx_tr, resto = train_test_split(idx, test_size=0.4, stratify=y, random_state=semilla)
    idx_va, idx_te = train_test_split(resto, test_size=0.5, stratify=y[resto], random_state=semilla)
    return idx_tr, idx_va, idx_te


def puerto_mas_frecuente_ataque(X_tr, y_tr):
    """Puerto de destino más común entre los ataques del conjunto de entrenamiento."""
    return int(X_tr.loc[y_tr == 1, "Destination Port"].value_counts().idxmax())


def _log1p_seguro(X):
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))


def modelo_svm():
    """SVM lineal con probabilidades calibradas (para poder usar umbrales entre 0 y 1).
    Pasos: log(1+x) -> estandarizar -> SVM lineal -> calibración sigmoide."""
    return Pipeline([
        ("log", FunctionTransformer(_log1p_seguro)),
        ("escala", StandardScaler()),
        ("svm", CalibratedClassifierCV(LinearSVC(C=1.0, dual=False, max_iter=5000),
                                       method="sigmoid", cv=3)),
    ])


def verificar_metricas(metricas):
    """Compara TU función metricas() con scikit-learn en un caso de prueba."""
    from sklearn.metrics import precision_score, recall_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = (rng.random(500) < 0.4).astype(int)
    try:
        m = metricas(y, p)
        ok = (abs(m["precision"] - precision_score(y, p)) < 1e-9
              and abs(m["recall"] - recall_score(y, p)) < 1e-9
              and m["tp"] + m["fp"] + m["fn"] + m["tn"] == 500)
        casos_borde = metricas(np.array([0, 0, 0]), np.array([0, 0, 0]))
        ok = ok and casos_borde["precision"] == 0.0 and casos_borde["recall"] == 0.0
    except NotImplementedError:
        raise
    except Exception as e:                        # noqa: BLE001
        print(f"   ✘ metricas() produjo un error: {e}")
        return False
    print("   ✔ metricas() coincide con scikit-learn" if ok else
          "   ✘ metricas() NO coincide con scikit-learn: revisa TP, FP, FN y las divisiones por cero")
    return ok


def simular_prevalencia(y, scores, umbral, prevalencias=(0.013, 0.005, 0.002, 0.001), semilla=1):
    """Rehace la evaluación como si hubiera MENOS ataques (conserva todos los benignos
    y submuestrea los ataques). Devuelve un DataFrame con precisión y recall."""
    rng = np.random.default_rng(semilla)
    ib, ia = np.where(y == 0)[0], np.where(y == 1)[0]
    filas = []
    for prev in prevalencias:
        n_a = max(int(len(ib) * prev / (1 - prev)), 1)
        sel = np.r_[ib, rng.choice(ia, min(n_a, len(ia)), replace=False)]
        pred = scores[sel] >= umbral
        real = y[sel] == 1
        tp, fp, fn = int((pred & real).sum()), int((pred & ~real).sum()), int((~pred & real).sum())
        filas.append({"prevalencia": prev, "ataques": int(real.sum()),
                      "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "fp": fp})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ gráficos
def _guardar(fig, ruta):
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    print(f"   [figura] {ruta}")


def graficar_clases(y, idx_tr, idx_va, idx_te, ruta, titulo):
    """Cantidad de benignos y ataques en cada partición."""
    nombres = ["Entrenamiento", "Validación", "Prueba"]
    ben = [int((y[i] == 0).sum()) for i in (idx_tr, idx_va, idx_te)]
    atq = [int((y[i] == 1).sum()) for i in (idx_tr, idx_va, idx_te)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    ax.bar(x - 0.2, ben, 0.4, color=AZUL, label="benigno")
    ax.bar(x + 0.2, atq, 0.4, color=ROJO, label="ataque")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    for i in range(3):
        ax.text(x[i] + 0.2, atq[i] * 1.15, f"{atq[i]:,}\n({atq[i] / (atq[i] + ben[i]) * 100:.2f} %)",
                ha="center", fontsize=8, color=ROJO)
        ax.text(x[i] - 0.2, ben[i] * 1.15, f"{ben[i]:,}", ha="center", fontsize=8, color=AZUL)
    ax.set_ylim(top=max(ben) * 8)
    ax.set_ylabel("Flujos (escala logarítmica)")
    ax.set_title(f"Partición estratificada — {titulo}")
    ax.legend()
    _guardar(fig, ruta)


def graficar_arbol(arbol, nombres_variables, ruta, titulo):
    """Dibuja el árbol poco profundo: cada camino hasta una hoja es una regla."""
    fig, ax = plt.subplots(figsize=(15, 6.5))
    plot_tree(arbol, feature_names=list(nombres_variables), class_names=["benigno", "ataque"],
              filled=True, rounded=True, fontsize=7, max_depth=3, ax=ax, impurity=False)
    ax.set_title(f"Reglas del árbol (profundidad ≤ 3) — {titulo}")
    _guardar(fig, ruta)


def graficar_comparacion(filas, ruta, titulo):
    """filas: lista de (nombre, precision, recall). Barras agrupadas."""
    nombres = [f[0] for f in filas]
    x = np.arange(len(filas))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - 0.2, [f[1] for f in filas], 0.4, color=AZUL, label="precisión")
    b2 = ax.bar(x + 0.2, [f[2] for f in filas], 0.4, color=ROJO, label="recall")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylim(0, 1.12)
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    _guardar(fig, ruta)


def graficar_barrido(tabla, ruta, titulo, umbral_marcado=None):
    """Precisión, recall y F1 en función del umbral + falsos positivos (eje derecho)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tabla["umbral"], tabla["precision"], "o-", color=AZUL, label="precisión")
    ax.plot(tabla["umbral"], tabla["recall"], "s-", color=ROJO, label="recall")
    ax.plot(tabla["umbral"], tabla["f1"], "^--", color=GRIS, label="F1")
    ax.set_xlabel("Umbral de decisión (se alerta si probabilidad ≥ umbral)")
    ax.set_ylabel("Valor de la métrica")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(tabla["umbral"], tabla["fp"], width=0.03, color=NARANJA, alpha=0.35, label="falsos positivos")
    ax2.set_ylabel("Falsos positivos (barras)", color=NARANJA)
    if umbral_marcado is not None:
        ax.axvline(umbral_marcado, color=VERDE, ls=":", lw=2)
        ax.text(umbral_marcado, 0.03, f" elegido = {umbral_marcado:.2f}", color=VERDE, fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left")
    ax.set_title(titulo)
    _guardar(fig, ruta)


def graficar_curvas_pr(y, curvas, ruta, titulo, puntos=None):
    """Curvas precisión-recall. curvas: {nombre: scores}. puntos: {nombre: (recall, precision)}.
    Izquierda: vista completa. Derecha: ZOOM a la zona alta (0,8 a 1,0), donde se ven
    las diferencias entre modelos cuando ambos son buenos."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
    colores = [AZUL, ROJO, VERDE, NARANJA]
    marcas = ["*", "D", "P", "X"]
    for k, ax in enumerate(axs):
        for (nombre, s), c in zip(curvas.items(), colores):
            p, r, _ = precision_recall_curve(y, s)
            ap = average_precision_score(y, s)
            ax.plot(r, p, color=c, lw=2, label=f"{nombre}  (AP = {ap:.3f})" if k == 0 else None)
        for (nombre, (r, p)), m in zip((puntos or {}).items(), marcas):
            ax.scatter([r], [p], s=110, marker=m, color="black", zorder=5, label=nombre if k == 0 else None)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precisión")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.legend(loc="lower left", fontsize=8)
            ax.set_title("Vista completa")
        else:
            ax.set_xlim(0.8, 1.005)
            ax.set_ylim(0.8, 1.005)
            ax.set_title("Zoom: recall y precisión entre 0,8 y 1,0")
    fig.suptitle(titulo)
    _guardar(fig, ruta)


def graficar_matriz(tp, fp, fn, tn, ruta, titulo):
    """Matriz de confusión con colores."""
    m = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(np.log10(m + 1), cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if m[i, j] > m.max() / 20 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["predice\nbenigno", "predice\nataque"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["real\nbenigno", "real\nataque"])
    ax.set_title(titulo, fontsize=10)
    _guardar(fig, ruta)


def graficar_prevalencia(tablas, ruta, titulo):
    """tablas: {nombre: DataFrame de simular_prevalencia}."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (nombre, t), c in zip(tablas.items(), [AZUL, ROJO, VERDE]):
        ax.plot(t["prevalencia"] * 100, t["precision"], "o-", color=c, label=f"precisión — {nombre}")
        ax.plot(t["prevalencia"] * 100, t["recall"], "s--", color=c, alpha=0.5, label=f"recall — {nombre}")
    ax.set_xscale("log")
    primera = next(iter(tablas.values()))
    ax.set_xticks(primera["prevalencia"] * 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} %"))
    ax.minorticks_off()
    ax.set_xlabel("Porcentaje de ataques en el tráfico (escala logarítmica)")
    ax.set_ylabel("Valor de la métrica")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(titulo)
    _guardar(fig, ruta)
