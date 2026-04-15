import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import os



np.random.seed(42)

def data_load(path: str):
    '''
    Carga el dataset

    parámetros: path -> Ruta del archivo CSV.
    return: DataFrame con los datos cargados. 
    '''
    df =  pd.read_csv(path)
    return df 

def clear_data(df: pd.DataFrame):
    '''
        Limpia el Dataset para el análisis

        Acciones realizadas:
        - Elimina columnas de id y de texto que no son útiles para el análisis.
        - Elimina variable de fecha 'last_review' con muchos nulos.
        - Reemplaza los valores nulos por 0 en.
        - Convertir columnas a tipo category
    '''
    df_limpio = df.copy()
    columnas_eliminar = ['id', 'name', 'host_id', 'host_name', 'last_review']
    df_limpio = df_limpio.drop(columns=[col for col in columnas_eliminar if col in df_limpio.columns])

    if "reviews_per_month" in df_limpio.columns:
        df_limpio["reviews_per_month"]  = df_limpio["reviews_per_month"].fillna(0)

    columnas_categoricas = [col for col in df_limpio.columns if df_limpio[col].dtype == 'object']
    for col in columnas_categoricas:
        df_limpio[col] = df_limpio[col].astype('category')

    return df_limpio

def tratar_outliers_iqr(df: pd.DataFrame, columna_objetiva: str,):
    '''
        Detecta y trata autliers de la variable objetivo usando el método IQR

        Parámetros:
        - df: Dataframe limpio
        - columna_objetiva: variable objetivo

        Retrun:
        - Dataframe sin outliers.
        - Diccionario con resumen del tratamiento.
    '''
    df_filtrado = df.copy()
    q1 =df_filtrado[columna_objetiva].quantile(0.25)
    q3 =df_filtrado[columna_objetiva].quantile(0.75)
    iqr = q3 -q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior= q3 + 1.5 *iqr

    n_antes = len(df_filtrado)

    df_filtrado = df_filtrado[
        (df_filtrado[columna_objetiva] >= limite_inferior)
        & (df_filtrado[columna_objetiva]<= limite_superior)
    ].copy()

    n_despues = len(df_filtrado)

    resumen = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "outliers_eliminados": n_antes - n_despues,
        "porcentaje_eliminado": ((n_antes - n_despues)/ n_antes) * 100,
    }

    return df_filtrado, resumen

def guardar_outliers(resumen_outliers:dict, ruta_txt: str):
    '''
        Guarda el resumen del tratamiento de outliers en un archivo txt.

        Parámetros:
        - resumen_outliers: diccionario con métricas del IQR.
        - ruta_txt: Ruta del archivo txt de salida.
    '''
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("--- TRATAMIENTO DE OUTLIERS (IQR) ---\n\n")
        for clave, valor in resumen_outliers.items():
            if isinstance(valor, float):
                f.write(f"{clave}: {valor:.4f}\n")
            else:
                f.write(f"{clave}: {valor}\n")
        
        f.write("\n--- INTERPRETACIÓN ---\n")
        if resumen_outliers["porcentaje_eliminado"] < 5:
            f.write("El porcentaje de outliers eliminado es bajo, por lo que su impacto es limitado.\n")
        else:
            f.write("El porcentaje de outliers eliminado es significativo y puede afectar al análisis y al modelado.\n")

def resumen_estructural(df_original: pd.DataFrame, df_limpio: pd.DataFrame,resumen_outliers: dict, ruta_txt):
    '''
        Guada un resumen estructural y de limpieza en un .txt

        Parámetros:
        - df_original: Dataframe original.
        - df_limpio: Dataframe limpio tras la limpieza.
        - resumen_outliers: resumen del tratamiento de outliers.
        - ruta_txt: Ruta del archivo txt de salida. 
    '''
    memoria = df_original.memory_usage(deep=True).sum() / (1024**2)
    pct_null = (df_original.isnull().mean() * 100).sort_values(ascending=False)

    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("----  RESUMEN ESTRUCTURAL DEL DATASET ---\n\n")
        f.write(f"Filas originales: {df_original.shape[0]} \n")
        f.write(f"Columnas originales: {df_original.shape[1]} \n")
        f.write(f"Tamaño en momeria mb: {memoria:.2f} MB \n\n")

        f.write("\n----TIPOS DE DATOS POR COLUMNA ----\n")
        for col,dtypes in df_original.dtypes.items():
            f.write(f"-{col}: {dtypes}\n")

        f.write("\n---- PORCENTAJES DE NULOS POR COLUMNAS ----\n")
        for col, pct in pct_null.items():
            f.write(f"-{col}: {pct:.2f}% \n")

        f.write("\n---- DECISIONES DE LIMPIEZA ----\n")
        f.write("-Eliminación de variables de id y texto que no son útiles como 'id', 'name', 'host_id', 'host_name'.\n")
        f.write("-Se ha eliminado la variable de fecha 'last_review' puesto que contenia muchos nulos.\n")
        f.write("-En la variable 'reviews_per_month' se sustituyen los nulos por 0.\n")
        f.write("-Converción de variables tipo texto a 'category' para un mejor rendimiento.\n")

        f.write("\n---- RESUMEN DATASET LIMPIO ----\n")
        f.write(f"Filas: {df_limpio.shape[0]}\n")
        f.write(f"Columnas: {df_limpio.shape[1]}\n")
        f.write(f"Tamaño en memoria MB: {df_limpio.memory_usage(deep=True).sum() / (1024**2)}\n")

        f.write("---- Porcentajes de nulos ---\n")
        pct_limpio = (df_limpio.isnull().mean() * 100).sort_values(ascending=False)
        for col, pct in pct_limpio.items():
            f.write(f"-{col}: {pct:.2f}% \n")
        

def calculo_estadistico_descriptivo( df: pd.DataFrame, columna_objetiva: str, ruta_csv: str):
    '''
        Calcula estadísticos descriptivos de variables numéricas y lo guarda.

        parámetros:
        - df: dataset limpio.
        - columna_objetiva: variable objetiva
        - ruta_csv: ruta de salida para el CSV.
    '''
    col_num = df.select_dtypes(include=[np.number]).columns.tolist()
    describe_df = df[col_num].describe().T
    moda = df[col_num].mode().iloc[0]
    varianza = df[col_num].var()

    describe_df['moda'] = moda
    describe_df['varianza'] = varianza

    target = df[columna_objetiva]
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)

    metricas_target = pd.DataFrame({
        "Q1_target": q1,
        "Q3_target": q3,
        "IQR_target": q3 - q1,
        "skewness_target": skew(target, bias=False),
        "kurtosis_target": kurtosis(target, fisher=True, bias=False)
    }, index=[columna_objetiva]).T

    resultado_final = pd.concat([describe_df, metricas_target], axis=0)
    resultado_final.to_csv(ruta_csv, encoding="utf-8")


def grafica_histogramas(df: pd.DataFrame, ruta_salida: str):
    '''
        Genera un histograma para cada variable numérica

        Parámetros:
        - df: Dataframe limpio
        - ruta_salida: ruta de la salida de la imagen
    '''
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = 3
    n_filas = (len(cols_num) + n_cols -1) // n_cols

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_filas, figsize=(18, 5 * n_filas))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols_num):
        dato = df[col].dropna()

        p1 = dato.quantile(0.01)
        p99 = dato.quantile(0.99)
        dato_recortado = dato[(dato >= p1) & (dato <= p99)]
        sns.histplot(dato_recortado, kde=True, ax=axes[i], bins=30)
        axes[i].set_title(f"Histograma de {col} (recorte 1%-99%)")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frecuencia")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()

def grafica_boxplots(df: pd.DataFrame, columna_objetiva: str, ruta_salida: str):
    '''
        Genera Boxplots de la variable objetivo, segmentado por cada variable categórica.

        Parámetros:
        - df: Dataframe limpio
        - columna_objetiva: variable objetivo
        - ruta_salida: ruta de la salida de la imagen
    '''
    cols_categ = df.select_dtypes(include=["category", "object"]).columns.to_list()
    
    n_cols = 2
    n_filas = (len(cols_categ) + n_cols -1) // n_cols

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_filas, figsize=(16, 5 * n_filas))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols_categ):
        categ_order = df[col].value_counts().index

        if len(categ_order) > 10:
            categ_order = categ_order[:10]
            df_plot = df[df[col].isin(categ_order)].copy()
            titulo = f"{columna_objetiva} por {col} (top 10 categorías)"
        else:
            df_plot = df.copy()
            titulo =f"{columna_objetiva} por {col}"

        sns.boxplot(data=df_plot, x=col, y=columna_objetiva, ax=axes[i], order=categ_order, showfliers=False)
        axes[i].set_title(titulo)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(columna_objetiva)
        axes[i].tick_params(axis="x", rotation=45)


        if(df_plot[columna_objetiva]> 0).all():
            axes[i].set_yscale("log")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()

def frecuencia_categorica(df: pd.DataFrame, ruta_salida: str):
    '''
        Guarda frecuencia absolutas y relativas de variables categóricas en un txt.

        Parámetros:
        - df: Dataframe limpio
        - columna_objetiva: variable objetivo
        - ruta_salida: ruta de la salida del txt
    '''

    cols_categ = df.select_dtypes(include=["category", "object"]).columns.to_list()

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("--- FRECUENCIA DE VARIABLES CATEGÓRICAS ---\n\n")
        for col in cols_categ:
            f.write(f"---{col}---\n")
            freq_abs = df[col].value_counts(dropna=False)
            freq_rel = df[col].value_counts(normalize=True, dropna=False) * 100

            for categoria in freq_abs.index:
                f.write(
                    f"{categoria}: frecuencia absoluta = {freq_abs[categoria]},"
                    f"frecuencia relativa = {freq_rel[categoria]:.2f}%\n"
                )
            categ_dom = freq_rel.idxmax()
            pct_dom = freq_rel.max()
            f.write(f"Categoría dominante: {categ_dom} ({pct_dom:.2f}%)\n")

            if pct_dom > 70:
                f.write("Posible desbalance importante en esta variable.\n\n")
            else:
                f.write("Destribución relativamente equilibrada.\n\n")


def grafico_categorica(df: pd.DataFrame, ruta_salida: str):
    '''
        Genera un gráfico de barras para cada variable categórica

        Parámetros:
        - df: Dataframe limpio
        - columna_objetiva: variable objetivo
        - ruta_salida: ruta de la salida de la imagen
    '''
    cols_categ = df.select_dtypes(include=["category", "object"]).columns.to_list()
    
    n_cols = 2
    n_filas = (len(cols_categ) + n_cols -1) // n_cols

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_filas, figsize=(16, 5 * n_filas))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols_categ):
        conteo =df[col].value_counts()

        if len(conteo) > 10:
            top_10 = conteo.head(10)
            resto = conteo.iloc[10:].sum()
            conteo_plot = pd.concat([top_10, pd.Series({"otros":resto})])
            conteo_plot = conteo_plot.sort_values(ascending=False)
            titulo = f"Frecuencia de {col} (top 10 + Otros)"
        else:
            conteo_plot = conteo
            titulo = f"Frecuencia de {col}"
        sns.barplot(x=conteo_plot.index.astype(str), y=conteo_plot.values, ax=axes[i])
        axes[i].set_title(titulo)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frecuencia")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()

def correlaciones(df: pd.DataFrame, columna_objetiva: str, ruta_png: str, ruta_txt: str):
    '''
        Calcula la correlación de Pearson, genera el heatmap y guarda resultados claves.

        Parámetros:
        - df: Dataframe limpio
        - columna_objetiva: variable objetivo
        - ruta_png: ruta de la salida de la imagen
        - ruta_txt: ruta de salida del txt
    '''

    corr = df.select_dtypes(include=(np.number)).corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de calor de correlaciones de Pearson")
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=300, bbox_inches="tight")
    plt.close()

    corr_target = corr[columna_objetiva].drop(labels=[columna_objetiva]).abs().sort_values(ascending=False)
    top_3 = corr_target.head(3)

    multicolinealdad = []
    columnas = corr.columns.tolist()

    for i in range(len(columnas)):
        for j in range(i +1, len(columnas)):
            valor = corr.iloc[i, j]
            if abs(valor) > 0.9:
                multicolinealdad.append((columnas[i], columnas[j], valor))

    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("--- TOP 3 VARIABLES MÁS CORRELACIONADAS CON LA VARIABLE OBJETIVO ---\n")
        for variable, valor in top_3.items():
            f.write(f"{variable}: {valor:.4f}\n")
        
        f.write("\n--- INTERPRETACIÓN ---\n")
        for variable, valor in top_3.items():
            if valor > 0.5:
                f.write(f"{variable}: correlación fuerte con la variable objetivo.\n")
            elif valor > 0.3:
                f.write(f"{variable}: correlación moderada con la variable objetivo.\n")
            else:
                f.write(f"{variable}: correlación débil con la variable objetivo.\n")
        
        f.write("\n--- POSIBLES CASOS DE MULTICOLINEALIDAD (|r| > 0.9) ---\n\n")
        if multicolinealdad:
            for var1, var2, valor in multicolinealdad:
                f.write(f"{var1} - {var2}: {valor:.4f}\n")
        else:
            f.write("No se detectaron pares con |r| > 0.9.\n")

def main():
    datos = "data/listings.csv"
    carpeta_salida = "output"
    columna_objetiva = "price"

    os.makedirs(carpeta_salida, exist_ok=True)

    df_original = data_load(datos)
    df_limpio = clear_data(df_original)
    df_limpio, resumen_outliers = tratar_outliers_iqr(df_limpio, columna_objetiva)

    calculo_estadistico_descriptivo(
        df_limpio, columna_objetiva, os.path.join(carpeta_salida, "ej1_descriptivo.csv")
    )
    
    resumen_estructural(
        df_original, df_limpio, resumen_outliers, os.path.join(carpeta_salida, "ej1_resumen.txt")
    )
    guardar_outliers(
        resumen_outliers, os.path.join(carpeta_salida, "ej1_outliers.txt")
    )
    grafica_histogramas(
        df_limpio, os.path.join(carpeta_salida, "ej1_histogramas.png"),
    )
    grafica_boxplots(
        df_limpio, columna_objetiva ,os.path.join(carpeta_salida, "ej1_boxplots.png"),
    )
    frecuencia_categorica(
        df_limpio, os.path.join(carpeta_salida, "ej1_frecuencia_categ.txt")
    )
    grafico_categorica(
        df_limpio, os.path.join(carpeta_salida, "ej1_categoricas.png")
    )
    correlaciones(
        df_limpio, columna_objetiva, 
        os.path.join(carpeta_salida, "ej1_heatmap_correlacion.png"),
        os.path.join(carpeta_salida, "ej1_correlaciones.txt"),
    )

if __name__ == "__main__":
    main()