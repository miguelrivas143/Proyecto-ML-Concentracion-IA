# Proyecto-ML-Concentración-IA
Reto de la Concentración de Inteligencia Artificial Periodo 1

Este repositorio contiene la nueva estructura del dataset, así como una implementación manual del algoritmo K-Nearest Neighbors desarrollado en Python.

# Nueva estructura del dataset

Para el entrenamiento del modelo, se realizó el proceso de Feature Engineering sobre el dataset "REHAB". El dataset original consistía en series temporales tridimensionales recolectadas mediante sensores y un guante de rehabilitación en pacientes post-ictus.

El nuevo dataset está estructurado de la siguiente manera:

1. Los datos originales, que estaban en una forma tridimensional dentro de archivos ".npy", fueron aplanados y transformados a un formato tabular bidimensional.
2. Se realizó el proceso de Feature Engineering: por cada uno de los 12 canales de los sensores, se calcularon 6 métricas estadísticas en el dominio del tiempo. Estas son: desviación estándar, valor máximo, valor mínimo, rango, asimetría y Root Mean Square.
3. Esto generó un total de 72 columnas de características predictivas.
4. Se agregó una última columna llamada "etiqueta", la cual contiene el número del ejercicio realizado.

Se descartó la métrica de la media y la mediana para poder garantizar un buen funcionamiento del modelo. Esto se debe a que los datos procesados ya habían sido normalizados con media 0. Finalmente, se eliminaron los valores nulos.


 

   
