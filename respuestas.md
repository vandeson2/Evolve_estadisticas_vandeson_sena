# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
He realizado un análisis estadísitico descriptivo sobre un dataset de Airbnb de Madrid. El objetivo ha sido comprender la estructura de los datos, identificar posibles problemas de calidad y analizar las relaciones entre variables.

En el análisis de variables categóriocas con alta cardinalidad, se han representado únicamente las 10 categoriás más frecuente. En los gráficos de frecuencias, el resto de categorias se ha agrupado en 'otros'para mejorar la legitibilidad.
En los boxplots, se han mostrado solo las 10 categorías principales para evitar mezclar distribución heterogéneas.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

He utilizado el dataset de Madrid Airbnb obtenido de Kaggle.

La variable objetivo seleccionada es **price**, que representa el precio por noche del alojamiento.

Tiene sentido aplicar la regresión sobre esta variable porque es una variable numeríca continua, está influenciada por múltiples factores como (ubicación, tipo de habitación, disponibilidad, etc) y es una variable de interés real en el contexto del negocio, ya que permite estimar precios de nuevos alojamientos.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

Distribución:
- **price**: muestra una distribución asimétric positiva, con una concentración de valores bajos y una cola larga hacia valores altos, lo que indica la presencia de alojamientos de precio elevado.
- **number_of_reviews** y **reviews_per_month**: presentan asimetría, con muchos valores cercanos a cero.
- **availability_365**: presenta una distribución más uniforme.

Se ha detectado outliers en la variable **price**, característicos en este tipo de dataset por los alojamientos de lujo.
Para su tratamiento se utilizó el método del rango intercuartílico (IQR):
- Q1= 35.0000
- Q3= 100.0000
- IQR= 65.0000
Se eliminaron un **9.7% de los dato**s, lo que representa un impacto moderado.
He elejido el método IQR frente al Z-score porque es más robusto ante distribuciones asimétricas como la observada en 'price' y no asume normalidad en los datos.
La eliminación de estos valores permite evitar que los modelos de regresión se vean distorcionados por valores extremos.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

Las tres variables numéricas con mayor correlación (en valor absoluto) con la variable objetiva 'precio' son:
- calculated_host_listings_count: r = 0.2052
- latitude: r = 0.0907
- availability_365: r = 0.0554

Por lo general las correlaciones observadas son **débiles**, lo que indica que el precio depende de múltiples factores y no de una sola variable. 
No detectaron problemas graves de multicolinealidad, por lo que las variables pueden utilizarse conjuntamente en modelos de regresión.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

Si, el dataset presenta valores nulos, principalmente en las variables:

- **last_review**: con un 28.73% se ha elimida la varible debido al alto porcentaje de nulos.
- **reviews_per_month**: con un porcentaje aproximadamente de 28,73%, se ha reemplazado por 0, ya que la ausencia de reviews puede interpretarse como falta de actividad

El resto de variables representa un porcentaje de nulos muy bajo o sin nulos, por lo que no ha sido necesario aplicar mas cambios.


**Conclusiones del análisis descriptivo**
El dataset presenta características típicas de datos reales, como distribuciones asímetricas y presencia de autliers en la variable objetivo.
El precio muestra una alta variabilidad y una distribución sesgada, lo que será necesario aplicar modelo robustos para su predicción.
Las correlaciones débiles indican que el precio depende de múltiples factores, lo que refuerza la necesidad de utilizar modelos multivariantes.
El tratamiento de valores nulos y outlieres permite mejorar la calidad del dataset y evitar que valores extremos o datos incompletos afecten negativamente al modelado.

## Ejercicio 2 — Inferencia con Scikit-Learn

En este ejercicio se ha aplicado un modelo de regresión lineal utilizando la librería Scikit-Learn con el objetivo de predecir la variable 'price' apartir del resto de variables del dataset. En primer lugar, se ha realizado un preprocesamiento de los datos, S e han eliminadovariables identificativas y textuales que no aportan valor predictivo, como 'id', 'name', 'host_id', 'host_name' y 'last_review'. Además, los valores nulos de 'reviews_por_month' se han reemplazado por 0.
Las variables categóricas ('neighbourhood', 'neighbourhood' y 'room_type') se han transformado mediante OneHotEncoder, mientras que las variables numéricas se han escalado utilizando StandardScaler. Este proceso permite que el modelo trate correctamente las variables y evita sesgo debido a diferentes escalas.
Posteriormente, el dataset se ha dividido en conjunto de entrenamiento y test con una proporción 80/20 y 'random_state=42'. 
A continuación, se ha entrenadi un modelo de regresión lineal mediante un Pipeline, integrandi el preprocesamiento y el modelo en un único flujo.
Finalmente , se ha evaluado el modelo utilizando las métricas MAE, RMSE y R2, lo que permite analizar su capacidad predicativa y detectar sus limitaciones.  

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

Los valores obtenidos para el modelo de regresión lineal sobre el conjunto de test son:
- **MAE:** 133.83
- **RMSE:** 596.98
- **R2:** 0.0147

El modelo no funciona bien, ya que presenta un rendimiento muy bajo.
El valor del MAE indica que el error medio es elevado en relación con los precios del dataset, mientras que el RMSE, significamente mayor, muestra que existen errores grandes en algunas predicciones.
Además, el valor de R2 es muy cercano a 0, lo que indica que el modelo apenas es capaz de explicar la variabilidad de la variable objetivo. En concreto, explica menos del 2% de la variación del precio.
Esto sugiere que la relación entre variables predictoras y el precio no es lineal o que existen factores relevantes que no están siendo capturados por el modelo.
En conclisión, la regresión lineal no es adecuada para este problema y seria necesario utilizar modelos más complejos para mejorar la capacidad predictiva.



---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
En este ejercicio se ha implementado desde cero un modelo de regresión lineal múltiple utilizando NumPy y la solución analitica de Mínimos Cuadrado Ordinarios.
El objetivo ha sido calcular los coeficientes del modelo mediante la fórmula mmatricial y evaluar su rendimiento utilizando métricas como MAE, RMSE y R2. Además, se ha analizado la capacidad del modelo para aproximar los coeficientes reales del sistema generendor de datos.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

La fórmula β = (XᵀX)⁻¹ Xᵀy es la forma directa para resolver una regresión lineal de un solo golpe. En lugar de entrenar el modelo por pasos, usa álgebra matricial para encontrar los coeficientes exactos que dan el menor error posible.
β representa los pesos o coeficientes óptimos. Es la combinación matemática que mejor se ajusta. 
La columna de unos es fundamental para estimar el interceptor. Sin esa columna, obligas al modelo a pasar siempre por (0,0), lo que arruinaría la prediccción en casi cualquier escenario real, ya que casi nunca se empieza por 0. Esta columna permite que la línea flote.


**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |                |
| β₁        | 2.0       |                |
| β₂        | -1.0      |                |
| β₃        | 0.5       |                |

  Intercepto (β₀): 4.864995
  β1 (feature 1): 2.063618
  β2 (feature 2): -1.117038
  β3 (feature 3): 0.438517

Coeficientes reales de referencia:
  Intercepto (β₀): 5.000000
  β1 (feature 1): 2.000000
  β2 (feature 2): -1.000000
  β3 (feature 3): 0.500000


**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

- MAE  : 1.166462
- RMSE : 1.461243
- R²   : 0.689672

- El valore del MAE hay un error medio de aproximadamente 1.17 unidades, lo que representa un buen ajuste.
- El RMSE, es ligeramente superior al MAE, indica que no existen errores extremadamente grandes.
- El R2 muestra que el modelo es capaz de explicar aproximadamente el 69% de la variabilidad de la varible objetivo. Aunque es un valor algo inferior al esperado.

Las métricas obtenidas se aproximan a las referencias del enunciado, lo que valida tanto la implementación del modelo como su capacidad para capturar la relación entre variables.


---

## Ejercicio 4 — Series Temporales
---
Se ha analizado una serie temporal sintética con datos diarios entre 2018 y 2023. A partir de su descomposición en tendencias, estacionalidad y residuo, se ha estudiado el comportamiento de cadacomponente.
La serie presenta una tendencia creciente, una estacionalidad anual bien definida y un residuo que se comportacomo ruido aleatorio. El análisis estadístico del residuo confirma que cumpe las propiedadesde un ruido gaussiano, lo que indica que la composición ha sido adecuada.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

Si, la serie presenta una tendencia creciente de tipo lineal. A lo largo del periodo analizado, los valores aumentan de forma progresiva desde aproximadamente 50 hasta valores cercanos a 170. Esto indica una tendencia positiva sostenida en el tiempo, con una magnitud significativa.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional
Si, la serie presenta una clara estacionalidad.
El periodo es aproximadamente de 365 días, lo que indica una estacionalidad anual. 
La amplitud del patrón estacional se sitúa aproximadamente entre -20 y +10 unidades, lo que supone una variación total cercana a 30 unidades.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

Si, se aprecian ciclos de largo plazo en la serie.
Estos ciclos manifestan como oscilaciones más suaves y de mayor duración que la estacionalidad, y se superponen a la tendebcua creciente. A diferencia de la tendencia, que es un cambio sistematico y sostenido en una dirección, los ciclos presentas subidas y bajadas a largo plazo sun una dirección fija.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

Si, el residuo se ajusta a un ruido ideal.
La media del residuo es aproximadamente 0.127, muy cercana a cero, y la desviación es de 3.22, lo que indica una variabilidad moderada y consistente con un ruido gaussiano.
El test de normalidad Jarque-Bera arroja un p-value de 0.576, po lo que no se rechaza la hipótesis de normalidad. Además, el análisis gráfico( hisograma, ACF y PACF) muestra que el residuo es simétrico, no presente autocorrelación y es estacionario.
En conjunto, estos resultados indican que el residuo se comporta como ruido blanco gaussiano.

---

*Fin del documento de respuestas*
