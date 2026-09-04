import os
import numpy as np
import pandas as pd
from scipy.stats import skew


ruta_carpeta = r'C:\Users\Miguel\OneDrive\Documentos\Semestre 3\Semestre 4\Semestre 5\Semestre 6\Semestre 7\Reto\REHAB\Rehab_exercise\d02_processed_data' #Ruta de la carpeta de los ejercicios con datos procesados
lista_dataframes = []

nombres_originales = ['pitch1', 'yaw1', 'roll1', 'pitch2', 'yaw2', 'roll2', 'f1', 'f2', 'f3', 'f4', 'f5', 'pitch3'] #Nombre de los canales de los sensores

metricas = ['std', 'max', 'min', 'rango', 'skew', 'rms'] #Metricas que utilizaremos para extraer nuevas características

for mov_id in range(16): #Vamos a iterar sobre cada uno de los movimietos
    mov_str = f"{mov_id:03d}"
    ruta_1 = os.path.join(ruta_carpeta, f"{mov_str}_1.npy") #Archivos que contienen los datos de cada movimiento captados con el sensor 1
    ruta_2 = os.path.join(ruta_carpeta, f"{mov_str}_2.npy") #Archivos que contienen los datos de cada movimiento captados con el sensor 2

    if os.path.exists(ruta_1) and os.path.exists(ruta_2):
    #Cargamos los archivos
        matriz_1 = np.load(ruta_1)
        matriz_2 = np.load(ruta_2)

        matriz_3d = np.concatenate((matriz_1, matriz_2), axis=2) #Se unen las matrices para tener los 12 canales juntos
        n_muestras, n_tiempo, n_canales = matriz_3d.shape

        datos_extraidos = []
        nombres_columnas = []
    #Hacemos los nombres de las 72 columnas
        for nombre in nombres_originales:
            for metrica in metricas:
                nombres_columnas.append(f'{nombre}_{metrica}')

        nombres_columnas.append('etiqueta')
    #Iteramos sobre cada muestra para extraer las características
        for i in range(n_muestras):
            muestra_actual = matriz_3d[i, :, :]
            caracteristicas_muestra = []
        #Iteramos sobre cada canal para extraer las características
            for canal in range(n_canales):
                senal_canal = muestra_actual[:, canal]

                std = np.std(senal_canal)
                v_max = np.max(senal_canal)
                v_min = np.min(senal_canal)
                rango = v_max - v_min
                asimetria = skew(senal_canal)
                rms = np.sqrt(np.mean(senal_canal**2))

                caracteristicas_muestra.extend([
                    std, v_max, v_min, rango, asimetria, rms])

            caracteristicas_muestra.append(mov_id)
            datos_extraidos.append(caracteristicas_muestra)
    #Se convierte la lista de datos extraidos en un DataFrame
        df_movimiento = pd.DataFrame(
             datos_extraidos,
            columns=nombres_columnas
        )
        lista_dataframes.append(df_movimiento)
#Unimos todos los DataFrames
dataset_final = pd.concat(lista_dataframes, ignore_index=True)
#Quitamos todos los valores nulos
dataset_final = dataset_final.dropna()

dataset_final.to_csv(
    r'C:\Users\Miguel\OneDrive\Documentos\Semestre 3\Semestre 4\Semestre 5\Semestre 6\Semestre 7\Reto\Ejercicios\Proyecto-ML-Concentracion-IA' + r'\dataset_final.csv',
    index=False
)
