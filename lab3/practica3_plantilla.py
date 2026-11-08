#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Práctica 3 — Reglas, árbol y SVM: precisión, recall y umbral  (PLANTILLA DEL ESTUDIANTE)
=======================================================================================

Qué haces en esta práctica (todo en un solo ejercicio):
  1. Partir los datos (60 % entrenamiento / 20 % validación / 20 % prueba).
  2. Evaluar una REGLA escrita a mano (puerto de destino).
  3. Entrenar un ÁRBOL DE DECISIÓN y leer sus reglas.
  4. Entrenar un SVM LINEAL (ya está armado en p3_utils.py).
  5. Implementar PRECISIÓN y RECALL a partir de TP, FP y FN.
  6. Mover el UMBRAL y ver cómo cambian precisión, recall y falsas alarmas.
  7. ELEGIR el umbral en validación y evaluarlo UNA sola vez en prueba.
  8. Repetir con otro archivo (botnet) y comparar modelos.

Uso:
    python practica3_plantilla.py --dataset web            # práctica principal
    python practica3_plantilla.py --dataset bot            # extensión
    python practica3_plantilla.py --dataset web --prevalencia   # parte opcional

Completa las funciones marcadas con TODO. Los pasos que dependan de una
función pendiente se omiten y el script te lo avisa.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier, export_text

import p3_utils as U

try:                                   # evita errores de codificación en la consola de Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                      # noqa: BLE001
    pass

UMBRALES = np.round(np.arange(0.05, 0.96, 0.05), 2)          # barrido para la tabla y el gráfico
UMBRALES_FINOS = np.round(np.linspace(0.01, 0.99, 99), 2)    # búsqueda del umbral elegido
RECALL_OBJETIVO = 0.95


# =====================================================================================
#  FUNCIONES QUE DEBES IMPLEMENTAR
# =====================================================================================
def metricas(y_real, y_pred):
    """TODO 1 — Calcula la matriz de confusión y las métricas a partir de TP, FP, FN y TN.

    y_real, y_pred: arreglos de 0 y 1 (1 = ataque).
    Devuelve un diccionario con las claves:
        tp, fp, fn, tn, precision, recall, f1
    Fórmulas:
        precisión = TP / (TP + FP)      recall = TP / (TP + FN)
        F1 = 2 * precisión * recall / (precisión + recall)
    Si un denominador es 0, la métrica vale 0.0 (no dejes que el programa falle).
    """
    # >>> ESCRIBE AQUÍ TU CÓDIGO (TODO 1) <<<
    # Pista: cuenta TP, FP, FN, TN y calcula precisión, recall y F1
    raise NotImplementedError("TODO 1: completa la función metricas()")


def evaluar_regla_puerto(X, y, puerto):
    """TODO 2 — Regla escrita a mano: «si Destination Port == puerto entonces ATAQUE».

    Devuelve el diccionario de metricas() para esa regla.
    """
    # >>> ESCRIBE AQUÍ TU CÓDIGO (TODO 2) <<<
    # Pista: predice 1 cuando el puerto de destino coincide y llama a metricas()
    raise NotImplementedError("TODO 2: completa la función evaluar_regla_puerto()")


def entrenar_arbol(X_tr, y_tr, profundidad=8, min_hoja=20):
    """TODO 3 — Entrena un DecisionTreeClassifier con max_depth=profundidad,
    min_samples_leaf=min_hoja y random_state=0. Devuelve el árbol ya entrenado.

    (min_samples_leaf obliga a que cada hoja tenga varias muestras; así el árbol
    produce varias probabilidades distintas y se puede mover el umbral.)
    """
    # >>> ESCRIBE AQUÍ TU CÓDIGO (TODO 3) <<<
    # Pista: crea el DecisionTreeClassifier y llama a fit()
    raise NotImplementedError("TODO 3: completa la función entrenar_arbol()")


def reglas_texto(X_tr, y_tr, profundidad=3):
    """TODO 4 — Entrena un árbol POCO profundo (max_depth=profundidad, min_samples_leaf=20,
    random_state=0) y devuelve la tupla (texto_de_reglas, arbol), donde el texto sale de
    sklearn.tree.export_text(arbol, feature_names=lista_de_columnas).
    Cada camino de la raíz a una hoja es una regla SI-ENTONCES.
    """
    # >>> ESCRIBE AQUÍ TU CÓDIGO (TODO 4) <<<
    # Pista: entrena un árbol de profundidad 3 y usa export_text()
    raise NotImplementedError("TODO 4: completa la función reglas_texto()")


def barrido_umbral(y, scores, umbrales):
    """TODO 5 — Para cada umbral t, predice ATAQUE cuando score >= t y calcula las métricas.

    Devuelve un DataFrame con una fila por umbral y las columnas:
        umbral, precision, recall, f1, fp, fn, alertas      (alertas = TP + FP)
    Usa TU función metricas().
    """
    # >>> ESCRIBE AQUÍ TU CÓDIGO (TODO 5) <<<
    # Pista: recorre los umbrales, llama a metricas() y arma el DataFrame
    raise NotImplementedError("TODO 5: completa la función barrido_umbral()")


def elegir_umbral(y, scores, criterio="recall_min", recall_min=RECALL_OBJETIVO,
                  umbrales=UMBRALES_FINOS):
    """TODO 6 — Elige un umbral con un criterio explícito. Devuelve (umbral, alcanzado).

    criterio = "recall_min": entre los umbrales con recall >= recall_min, el de MAYOR precisión.
    criterio = "f1":         el umbral con MAYOR F1.
    Empates: elige el umbral más alto (menos alertas).
    Si ningún umbral cumple recall_min, devuelve el umbral de mayor recall y alcanzado=False.
    Sugerencia: reutiliza barrido_umbral() con la lista de umbrales finos.
    """
    # >>> ESCRIBE AQUÍ TU CÓDIGO (TODO 6) <<<
    # Pista: filtra la tabla del barrido según el criterio y devuelve (umbral, True/False)
    raise NotImplementedError("TODO 6: completa la función elegir_umbral()")


# =====================================================================================
#  A PARTIR DE AQUÍ NO ES NECESARIO MODIFICAR (flujo de la práctica y gráficos)
# =====================================================================================
def paso(titulo, funcion, E):
    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)
    try:
        funcion(E)
    except NotImplementedError as e:
        print(f"[PENDIENTE] {e}")
    except KeyError as e:
        print(f"[OMITIDO] depende de un paso o función pendiente: {e}")


def mostrar(tabla, cols=None):
    t = tabla if cols is None else tabla[cols]
    print(t.round(4).to_string(index=False))


def paso0_verificacion(E):
    print("Comprobando que metricas() está bien implementada...")
    U.verificar_metricas(metricas)


def paso1_datos(E):
    X, y = E["X"], E["y"]
    E["idx_tr"], E["idx_va"], E["idx_te"] = U.particion(y)
    tr, va, te = E["idx_tr"], E["idx_va"], E["idx_te"]
    print(f"Archivo: {U.TITULOS[E['nombre']]}")
    print(f"Flujos: {len(y):,}   Características: {X.shape[1]}   Ataques: {int(y.sum()):,} ({y.mean() * 100:.3f} %)")
    for nom, idx in (("Entrenamiento", tr), ("Validación", va), ("Prueba", te)):
        print(f"   {nom:14s} {len(idx):>8,} flujos   {int(y[idx].sum()):>5,} ataques")
    U.graficar_clases(y, tr, va, te, os.path.join(E["figs"], f"{E['nombre']}_01_clases.png"), U.TITULOS[E["nombre"]])
    E["X_tr"], E["y_tr"] = X.iloc[tr], y[tr]
    E["X_va"], E["y_va"] = X.iloc[va], y[va]
    E["X_te"], E["y_te"] = X.iloc[te], y[te]


def paso2_regla(E):
    puerto = U.puerto_mas_frecuente_ataque(E["X_tr"], E["y_tr"])
    m = evaluar_regla_puerto(E["X_va"], E["y_va"], puerto)
    E["puerto"], E["m_regla_va"] = puerto, m
    print(f"Regla: SI Destination Port == {puerto} ENTONCES ataque   (evaluada en VALIDACIÓN)")
    print(f"   TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']:,}")
    print(f"   precisión={m['precision']:.3f}   recall={m['recall']:.3f}   F1={m['f1']:.3f}")


def paso3_arbol(E):
    arbol = entrenar_arbol(E["X_tr"], E["y_tr"])
    E["arbol"] = arbol
    E["s_arbol_va"] = arbol.predict_proba(E["X_va"])[:, 1]
    m = metricas(E["y_va"], (E["s_arbol_va"] >= 0.5).astype(int))
    E["m_arbol_va"] = m
    print(f"Árbol (profundidad 8, mín. 20 por hoja): probabilidades distintas = {len(np.unique(E['s_arbol_va']))}")
    print(f"   Con umbral 0,5 en validación: precisión={m['precision']:.3f}  recall={m['recall']:.3f}  FP={m['fp']}  FN={m['fn']}")
    texto, arbol3 = reglas_texto(E["X_tr"], E["y_tr"])
    print("\nReglas del árbol poco profundo (class 1 = ataque):\n")
    print(texto)
    U.graficar_arbol(arbol3, E["X_tr"].columns, os.path.join(E["figs"], f"{E['nombre']}_02_arbol_reglas.png"),
                     U.TITULOS[E["nombre"]])


def paso4_svm(E):
    svm = U.modelo_svm().fit(E["X_tr"], E["y_tr"])
    E["svm"] = svm
    E["s_svm_va"] = svm.predict_proba(E["X_va"])[:, 1]
    m = metricas(E["y_va"], (E["s_svm_va"] >= 0.5).astype(int))
    E["m_svm_va"] = m
    print(f"SVM lineal calibrado: probabilidades distintas = {len(np.unique(E['s_svm_va'])):,}")
    print(f"   Con umbral 0,5 en validación: precisión={m['precision']:.3f}  recall={m['recall']:.3f}  FP={m['fp']}  FN={m['fn']}")


def paso5_barrido(E):
    y_va, nombre, figs = E["y_va"], E["nombre"], E["figs"]
    for clave, etiqueta in (("s_svm_va", "SVM lineal"), ("s_arbol_va", "Árbol")):
        tabla = barrido_umbral(y_va, E[clave], UMBRALES)
        E["barrido_" + clave] = tabla
        print(f"\n--- {etiqueta}: efecto del umbral (conjunto de VALIDACIÓN) ---")
        mostrar(tabla)
        tabla.to_csv(os.path.join(figs, f"{nombre}_barrido_{clave[2:-3]}.csv"), index=False)
        ok = bool((np.diff(tabla["recall"].to_numpy()) <= 1e-12).all())
        print("   " + ("✔" if ok else "✘") + " comprobación: el recall nunca aumenta al subir el umbral")
        U.graficar_barrido(tabla, os.path.join(figs, f"{nombre}_03_barrido_{clave[2:-3]}.png"),
                           f"{etiqueta}: precisión y recall según el umbral — {U.TITULOS[nombre]}")
    puntos = {f"Regla puerto {E['puerto']}": (E["m_regla_va"]["recall"], E["m_regla_va"]["precision"])}
    U.graficar_curvas_pr(y_va, {"SVM lineal": E["s_svm_va"], "Árbol": E["s_arbol_va"]},
                         os.path.join(figs, f"{nombre}_04_curvas_pr.png"),
                         f"Curvas precisión-recall (validación) — {U.TITULOS[nombre]}", puntos)
    U.graficar_comparacion(
        [(f"Regla\npuerto {E['puerto']}", E["m_regla_va"]["precision"], E["m_regla_va"]["recall"]),
         ("Árbol\n(umbral 0,5)", E["m_arbol_va"]["precision"], E["m_arbol_va"]["recall"]),
         ("SVM\n(umbral 0,5)", E["m_svm_va"]["precision"], E["m_svm_va"]["recall"])],
        os.path.join(figs, f"{nombre}_05_comparacion_umbral05.png"),
        f"Regla frente a modelos con umbral 0,5 (validación) — {U.TITULOS[nombre]}")


def paso6_umbral(E):
    y_va = E["y_va"]
    filas = []
    for clave, etiqueta in (("s_svm_va", "SVM lineal"), ("s_arbol_va", "Árbol")):
        for criterio, desc in (("recall_min", f"recall ≥ {RECALL_OBJETIVO}"), ("f1", "máximo F1")):
            t, ok = elegir_umbral(y_va, E[clave], criterio)
            m = metricas(y_va, (E[clave] >= t).astype(int))
            E[f"t_{clave[2:-3]}_{criterio}"] = t
            filas.append({"modelo": etiqueta, "criterio": desc, "umbral": t, "criterio_cumplido": "sí" if ok else "NO",
                          "precision": m["precision"], "recall": m["recall"], "fp": m["fp"], "fn": m["fn"]})
    tabla = pd.DataFrame(filas)
    print("Umbral elegido en VALIDACIÓN según cada criterio:")
    mostrar(tabla)
    if (tabla["criterio_cumplido"] == "NO").any():
        print("\n⚠ En alguna fila ningún umbral alcanza el recall pedido: se muestra el de mayor recall.")
    # marcar el umbral elegido en los gráficos del barrido
    for clave, etiqueta in (("s_svm_va", "SVM lineal"), ("s_arbol_va", "Árbol")):
        U.graficar_barrido(E["barrido_" + clave],
                           os.path.join(E["figs"], f"{E['nombre']}_03_barrido_{clave[2:-3]}.png"),
                           f"{etiqueta}: precisión y recall según el umbral — {U.TITULOS[E['nombre']]}",
                           umbral_marcado=E[f"t_{clave[2:-3]}_recall_min"])


def paso7_prueba(E):
    y_te, nombre, figs = E["y_te"], E["nombre"], E["figs"]
    s_svm = E["svm"].predict_proba(E["X_te"])[:, 1]
    s_arb = E["arbol"].predict_proba(E["X_te"])[:, 1]
    E["s_svm_te"], E["s_arb_te"] = s_svm, s_arb
    m_regla = evaluar_regla_puerto(E["X_te"], y_te, E["puerto"])
    filas = [{"detector": f"Regla puerto {E['puerto']}", "umbral": "-", **m_regla}]
    for etiqueta, s, tag in (("Árbol", s_arb, "arbol"), ("SVM lineal", s_svm, "svm")):
        for desc, t in (("umbral 0,5", 0.5), (f"recall ≥ {RECALL_OBJETIVO}", E[f"t_{tag}_recall_min"]),
                        ("máximo F1", E[f"t_{tag}_f1"])):
            filas.append({"detector": f"{etiqueta} ({desc})", "umbral": round(t, 2),
                          **metricas(y_te, (s >= t).astype(int))})
    tabla = pd.DataFrame(filas)
    print("RESULTADO FINAL EN PRUEBA (se mira una sola vez; el umbral se eligió en validación):")
    mostrar(tabla, ["detector", "umbral", "precision", "recall", "f1", "fp", "fn"])
    tabla.to_csv(os.path.join(figs, f"{nombre}_resultados_prueba.csv"), index=False)
    print(f"\nPR-AUC (average precision) en prueba:  SVM = {average_precision_score(y_te, s_svm):.4f}   "
          f"Árbol = {average_precision_score(y_te, s_arb):.4f}")
    for etiqueta, tag, s in (("SVM lineal", "svm", s_svm), ("Árbol", "arbol", s_arb)):
        m = metricas(y_te, (s >= E[f"t_{tag}_recall_min"]).astype(int))
        U.graficar_matriz(m["tp"], m["fp"], m["fn"], m["tn"],
                          os.path.join(figs, f"{nombre}_06_matriz_{tag}.png"),
                          f"{etiqueta}, umbral {E[f't_{tag}_recall_min']:.2f} (prueba)")
    puntos = {f"Regla puerto {E['puerto']}": (m_regla["recall"], m_regla["precision"])}
    U.graficar_curvas_pr(y_te, {"SVM lineal": s_svm, "Árbol": s_arb},
                         os.path.join(figs, f"{nombre}_07_curvas_pr_prueba.png"),
                         f"Curvas precisión-recall (prueba) — {U.TITULOS[nombre]}", puntos)


def paso8_prevalencia(E):
    if not E["prevalencia"]:
        print("Parte opcional: ejecuta con la opción --prevalencia para activarla.")
        return
    tablas = {}
    for etiqueta, tag, s in (("SVM lineal", "svm", E["s_svm_te"]), ("Árbol", "arbol", E["s_arb_te"])):
        t = E[f"t_{tag}_recall_min"]
        tablas[etiqueta] = U.simular_prevalencia(E["y_te"], s, t)
        print(f"\n{etiqueta} con umbral {t:.2f} (el mismo, sin reajustar):")
        mostrar(tablas[etiqueta])
    U.graficar_prevalencia(tablas, os.path.join(E["figs"], f"{E['nombre']}_08_prevalencia.png"),
                           f"Efecto de la proporción de ataques — {U.TITULOS[E['nombre']]}")


def main():
    ap = argparse.ArgumentParser(description="Práctica 3: reglas, árbol, SVM, precisión, recall y umbral")
    ap.add_argument("--dataset", choices=["web", "bot", "ddos"], default="web")
    ap.add_argument("--datos", default="datos_limpios", help="carpeta con los archivos limpios")
    ap.add_argument("--figuras", default="figuras", help="carpeta donde se guardan gráficos y tablas")
    ap.add_argument("--prevalencia", action="store_true", help="activa la parte opcional (prevalencia)")
    a = ap.parse_args()

    os.makedirs(a.figuras, exist_ok=True)
    df = U.cargar_limpio(a.dataset, a.datos)
    X, y = U.separar_xy(df)
    E = {"nombre": a.dataset, "X": X, "y": y, "figs": a.figuras, "prevalencia": a.prevalencia}

    paso("PASO 0. Verificación de tu función metricas()", paso0_verificacion, E)
    paso("PASO 1. Datos y partición 60 / 20 / 20", paso1_datos, E)
    paso("PASO 2. Una regla escrita a mano", paso2_regla, E)
    paso("PASO 3. Árbol de decisión y sus reglas", paso3_arbol, E)
    paso("PASO 4. SVM lineal con probabilidades", paso4_svm, E)
    paso("PASO 5. ¿Qué pasa al mover el umbral? (validación)", paso5_barrido, E)
    paso("PASO 6. Elegir el umbral con un criterio", paso6_umbral, E)
    paso("PASO 7. Evaluación final en PRUEBA", paso7_prueba, E)
    paso("PASO 8 (opcional). Efecto de la proporción de ataques", paso8_prevalencia, E)
    print(f"\nListo. Revisa la carpeta '{a.figuras}/' (gráficos PNG y tablas CSV).")


if __name__ == "__main__":
    main()
