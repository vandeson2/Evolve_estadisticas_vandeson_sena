import pandas as pd
import numpy as np
import math
'''Calcula la media de una lista de números'''
def media(list_datos: list) -> float:
    return round(sum(list_datos) / len(list_datos), 2)

'''Calcula a mediana de una lista de números'''
def mediana(list_datos: list):
  if not list_datos: return None

  datos_ord = sorted(list_datos)
  n = len(datos_ord)

  if n % 2 == 0:
       return (datos_ord[n//2 -1] + datos_ord[n//2]) / 2
  else:
      return datos_ord[n//2]


'''Calcula el percentil de una lista de número'''
def percentil(list_datos: list, k: float):
    if not list_datos: return None
    if  k < 0 or k > 100: return 'Error: K debe estar entre 0 y 100.'

    datos_ord = sorted(list_datos)
    n = len(datos_ord)

    indice_virtual = (k / 100) * (n - 1)
    indice_bajo = int(indice_virtual)
    indice_alto = indice_bajo + 1
    residuo = indice_virtual - indice_bajo 

    if indice_alto >=  n:
        return datos_ord[n-1]
    else:
        valor_bajo = datos_ord[indice_bajo]
        valor_alto = datos_ord[indice_alto]
        resultado = valor_bajo + (residuo * (valor_alto - valor_bajo))

        return resultado
    
'''Calcula la varianza en una lista de números'''
def varianza(list_datos: list):
    n = len(list_datos)
    if n < 1: return None

    media = sum(list_datos) / n
    sum_cuadrado = sum((num - media) ** 2 for num in list_datos)

    return round(sum_cuadrado / n, 2)

'''Calvuro de la desviación'''
def desviacion(list_datos: list):
    v = varianza(list_datos)
    return round(math.sqrt(v), 2)

def iqr(list_datos: list):
    q3 = percentil(list_datos, 75)
    q1 = percentil(list_datos, 25)
    return round(q3 - q1, 2)
if __name__ == "__main__":
    np.random.seed(42)
    edad = list(np.random.randint(28, 68, 100))
    salario = list(np.random.normal(45000, 15000, 100))
    experiencia = list(np.random.randint(0, 30, 100))

    np.random.seed(42)

    df = pd.DataFrame({
        'edad': np.random.randint(28, 68, 100),
        'salario': np.random.normal(45000, 15000, 100),
        'experiencia': np.random.randint(0, 30, 100)
    })
    ## hacerlo con todas las funciones que se han creado
    print("Resultado pandas: ")
    print("---------------------------")
    print(df.describe())

    print("Resultado edad: ")
    print("---------------------------")
    print(media(edad))
    print(mediana(edad))
    print(percentil(edad, 75))
    print(varianza(edad))
    print(desviacion(edad))
    print(iqr(edad))

    print("Resultado salario: ")
    print("---------------------------")
    print(media(salario))
    print(mediana(salario))
    print(percentil(salario, 75))
    print(varianza(salario))
    print(desviacion(salario))
    print(iqr(salario))
    
    print("Resultado experiencia: ")
    print("---------------------------")
    print(media(experiencia))
    print(mediana(experiencia))
    print(percentil(experiencia, 75))
    print(varianza(experiencia))
    print(desviacion(experiencia))
    print(iqr(experiencia))
