"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 4
Análisis y Descomposición de Series Temporales
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio trabajarás con una serie temporal sintética generada con
una semilla fija. Tendrás que:

  1. Visualizar la serie completa.
  2. Descomponerla en sus componentes: Tendencia, Estacionalidad y Residuo.
  3. Analizar cada componente y responder las preguntas del fichero
     Respuestas.md (sección Ejercicio 4).
  4. Evaluar si el ruido (residuo) se ajusta a un ruido ideal (gaussiano
     con media ≈ 0 y varianza constante).

LIBRERÍAS PERMITIDAS
--------------------
  - numpy, pandas
  - matplotlib, seaborn
  - statsmodels   (para seasonal_decompose y adfuller)
  - scipy.stats   (para el test de normalidad del ruido)

SALIDAS ESPERADAS (carpeta output/)
------------------------------------
  - output/ej4_serie_original.png      → Gráfico de la serie completa
  - output/ej4_descomposicion.png      → Los 4 subgráficos de descomposición
  - output/ej4_acf_pacf.png           → Gráfico ACF y PACF del residuo
  - output/ej4_histograma_ruido.png   → Histograma + curva normal del residuo
  - output/ej4_analisis.txt            → Estadísticos numéricos del análisis

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import jarque_bera, norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)


# =============================================================================
# GENERACIÓN DE LA SERIE TEMPORAL SINTÉTICA — NO MODIFICAR ESTE BLOQUE
# =============================================================================

def generar_serie_temporal(semilla=42):
    """
    Genera una serie temporal sintética con componentes conocidos.

    La serie tiene:
      - Una tendencia lineal creciente.
      - Estacionalidad anual (periodo 365 días).
      - Ciclos de largo plazo (periodo ~4 años).
      - Ruido gaussiano.

    Parámetros
    ----------
    semilla : int — Semilla aleatoria para reproducibilidad (NO modificar)

    Retorna
    -------
    serie : pd.Series con índice DatetimeIndex diario (2018-01-01 → 2023-12-31)
    """
    rng = np.random.default_rng(semilla)

    # Índice temporal: 6 años de datos diarios
    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    # --- Componentes ---
    # 1. Tendencia lineal
    tendencia = 0.05 * t + 50

    # 2. Estacionalidad anual (periodo = 365.25 días)
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365.25) \
                   +  6 * np.cos(4 * np.pi * t / 365.25)

    # 3. Ciclo de largo plazo (periodo ~ 4 años = 1461 días)
    ciclo = 8 * np.sin(2 * np.pi * t / 1461)

    # 4. Ruido gaussiano
    ruido = rng.normal(loc=0, scale=3.5, size=n)

    # Serie completa (modelo aditivo)
    valores = tendencia + estacionalidad + ciclo + ruido

    serie = pd.Series(valores, index=fechas, name="valor")
    return serie


# =============================================================================
# TAREA 1 — Visualizar la serie completa
# =============================================================================

def visualizar_serie(serie):
    """
    Genera y guarda un gráfico de la serie temporal completa.

    Salida: output/ej4_serie_original.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal a visualizar

    Pistas
    ------
    - Usa fig, ax = plt.subplots(figsize=(14, 4))
    - Añade título, etiquetas de ejes y una cuadrícula suave
    - Guarda con plt.savefig("output/ej4_serie_original.png", dpi=150, bbox_inches='tight')
    """

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(serie.index, serie.values, linewidth=1, label="Serie Temporal")
    ax.set_title("Visualización de la Serie Temporal (2018-2023)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("output/ej4_serie_original.png", dpi=150, bbox_inches="tight")


    


# =============================================================================
# TAREA 2 — Descomposición de la serie
# =============================================================================

def descomponer_serie(serie):
    """
    Descompone la serie en Tendencia, Estacionalidad y Residuo usando
    statsmodels.tsa.seasonal.seasonal_decompose y guarda el gráfico.

    Salida: output/ej4_descomposicion.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal

    Retorna
    -------
    resultado : DecomposeResult — Objeto con atributos .trend, .seasonal, .resid

    Pistas
    ------
    - from statsmodels.tsa.seasonal import seasonal_decompose
    - Usa model='additive' y period=365
    - resultado.plot() genera los 4 subgráficos automáticamente
    - Guarda la figura con fig.savefig(...)
    """
    # TODO: Implementa la descomposición
    # resultado = seasonal_decompose(...)
    # fig = resultado.plot()
    # ...
    resultado = seasonal_decompose(serie, model='additive', period=365)
    fig = resultado.plot()
    fig.set_size_inches(12, 10)
    fig.savefig("output/ej4_descomposicion.png", dpi=150, bbox_inches='tight')

    return resultado

# =============================================================================
# TAREA 3 — Análisis del residuo (ruido)
# =============================================================================

def analizar_residuo(residuo):
    """
    Analiza el componente de residuo para determinar si se parece
    a un ruido ideal (gaussiano, media ≈ 0, varianza constante, sin autocorr.).

    Genera:
      - output/ej4_acf_pacf.png          → ACF y PACF del residuo
      - output/ej4_histograma_ruido.png  → Histograma + curva normal ajustada
      - output/ej4_analisis.txt          → Estadísticos numéricos

    Parámetros
    ----------
    residuo : pd.Series — Componente de residuo de la descomposición

    Pistas para el análisis numérico
    ----------------------------------
    - Media:     residuo.mean()
    - Std:       residuo.std()
    - Asimetría: residuo.skew()       (ruido ideal ≈ 0)
    - Curtosis:  residuo.kurtosis()   (ruido ideal ≈ 0 en exceso)
    - Test de normalidad Shapiro-Wilk o Jarque-Bera:
        from scipy.stats import jarque_bera
        stat, p = jarque_bera(residuo.dropna())
        Si p > 0.05 → no rechazamos normalidad
    - ADF Test para verificar estacionariedad:
        from statsmodels.tsa.stattools import adfuller
        resultado_adf = adfuller(residuo.dropna())
        p_adf = resultado_adf[1]
    - ACF / PACF:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    """
    # TODO: Limpia el residuo (elimina NaN al inicio/fin)
    residuo_limpio = residuo.dropna()

    # TODO: Calcula estadísticos básicos
    media    = residuo_limpio.mean()
    std      = residuo_limpio.std()
    asimetria = residuo_limpio.skew()
    curtosis  = residuo_limpio.kurtosis()


    # TODO: Test de estacionariedad (ADF)
    resultado_adf = adfuller(residuo_limpio, maxlag=20, autolag="AIC")
    adf_stat = resultado_adf[0]
    p_adf = resultado_adf[1]

    jb_stat, p_jb = jarque_bera(residuo_limpio)

    # TODO: Gráfico ACF y PACF del residuo → output/ej4_acf_pacf.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(residuo_limpio, ax=axes[0])
    axes[0].set_title("ACF del Residuo")
    plot_pacf(residuo_limpio, ax=axes[1])
    axes[1].set_title("PACF del Residuo")
    plt.tight_layout()
    plt.savefig("output/ej4_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()

    # TODO: Histograma del residuo con curva normal superpuesta
    # → output/ej4_histograma_ruido.png
    # Pista: usa scipy.stats.norm.pdf para la curva teórica
    plt.figure(figsize=(8, 5))
    plt.hist(residuo_limpio, bins=30, density=True, alpha=0.6)

    x = np.linspace(residuo_limpio.min(), residuo_limpio.max(), 200)
    y = norm.pdf(x, loc=media, scale=std)
    plt.plot(x, y, linewidth=2)

    plt.title("Histograma del Residuo con Curva Normal")
    plt.xlabel("Residuo")
    plt.ylabel("Densidad")
    plt.tight_layout()
    plt.savefig("output/ej4_histograma_ruido.png", dpi=150, bbox_inches="tight")
    plt.close()

    # TODO: Guardar estadísticos en output/ej4_analisis.txt
    with open("output/ej4_analisis.txt", "w", encoding="utf-8") as f:
        f.write("ANÁLISIS DEL RESIDUO\n\n")
        f.write(f"Media: {media:.6f}\n")
        f.write(f"Desviación típica: {std:.6f}\n")
        f.write(f"Asimetría: {asimetria:.6f}\n")
        f.write(f"Curtosis: {curtosis:.6f}\n")
       
        f.write(f"Jarque-Bera estadístico: {jb_stat:.6f}\n")
        f.write(f"Jarque-Bera p-value: {p_jb:.6f}\n")
        if p_jb > 0.5:
            f.write("Normalidad: no se rechaza la hipóteis de normalidad.\n")
        else:
           f.write("Normalidad: se rechaza la hipóteis de normalidad.\n") 
        
        f.write(f"ADF estadístico: {adf_stat:.6f}\n")
        f.write(f"ADF p-value: {p_adf:.6f}\n")
        if p_adf < 0.05:
            f.write("Estacionariedad: el residuo puede considerarse estacionario.\n")
        else:
            f.write("Estacionariedad: no se puede afirmar que el residuo sea estacionario.\n")


# =============================================================================
# MAIN — Ejecuta el pipeline completo
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("EJERCICIO 4 — Análisis de Series Temporales")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Paso 1: Generar la serie (NO modificar la semilla)
    # ------------------------------------------------------------------
    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)

    print(f"\nSerie generada:")
    print(f"  Periodo:      {serie.index[0].date()} → {serie.index[-1].date()}")
    print(f"  Observaciones: {len(serie)}")
    print(f"  Media:         {serie.mean():.2f}")
    print(f"  Std:           {serie.std():.2f}")
    print(f"  Min / Max:     {serie.min():.2f} / {serie.max():.2f}")

    # ------------------------------------------------------------------
    # Paso 2: Visualizar la serie completa
    # ------------------------------------------------------------------
    print("\n[1/3] Visualizando la serie original...")
    visualizar_serie(serie)

    # ------------------------------------------------------------------
    # Paso 3: Descomponer
    # ------------------------------------------------------------------
    print("[2/3] Descomponiendo la serie...")
    resultado = descomponer_serie(serie)

    # ------------------------------------------------------------------
    # Paso 4: Analizar el residuo
    # ------------------------------------------------------------------
    print("[3/3] Analizando el residuo...")
    if resultado is not None:
        analizar_residuo(resultado.resid)

    # ------------------------------------------------------------------
    # Resumen de salidas esperadas
    # ------------------------------------------------------------------
    print("\nSalidas esperadas en output/:")
    salidas = [
        "ej4_serie_original.png",
        "ej4_descomposicion.png",
        "ej4_acf_pacf.png",
        "ej4_histograma_ruido.png",
        "ej4_analisis.txt",
    ]
    for s in salidas:
        existe = os.path.exists(f"output/{s}")
        estado = "✓" if existe else "✗ (pendiente)"
        print(f"  [{estado}] output/{s}")

    print("\n¡Recuerda completar las respuestas en Respuestas.md!")
