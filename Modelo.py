import numpy as np
import pandas as pd

from KNN import KNN_Manual

ruta_csv = r'dataset_final.csv'
df = pd.read_csv(ruta_csv)

X = df.drop('etiqueta', axis=1).values #Seran mis caracteristica y la etiqueta es para predecir por eso la quitamos
y = df['etiqueta'].values #Etiqueta que queremos predecir
#Debido a que KNN usa la distancia Euclidiana, debemos de normalizar para que una característica no tenga más peso
media = np.mean(X, axis=0) 
desviacion = np.std(X, axis=0)
desviacion[desviacion == 0] = 1
X_norm = (X - media) / desviacion

np.random.seed(42) #Aseguramos que sea reproducible
indices = np.random.permutation(len(X))
corte = int(len(X) * 0.7) # 70% de los datos serán para entrenamiento y el resto de prueba

X_train, y_train = X_norm[indices[:corte]], y[indices[:corte]] #Separamos los datos de entrenamiento y prueba
X_test, y_test = X_norm[indices[corte:]], y[indices[corte:]]


modelo = KNN_Manual(k=5) #Ponemos que nuestro modelo utilice un KNN con 5 vecinos para predecir
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test) #Se hacen las predicciones con los datos de prueba

aciertos = np.sum(predicciones == y_test) #Contamos cuantas predicciones son correctas
precision = aciertos / len(y_test)#Se calula la precisión
print(f"Precisión: {precision * 100:.2f}%") 

#Se toman muestras aleatorias del conjunto de prueba para ver como se comparta el modelo
np.random.seed(42)
indices_muestra = np.random.choice(len(X_test), 5, replace=False)

#El modelo hace sus predicciones sobre las muestras
predicciones = modelo.predict(X_test[indices_muestra])
reales = y_test[indices_muestra]

#Se imprimen los resultados
for i in range(5):
    predicho = predicciones[i]
    real = reales[i]
            
    print(f"Predicción: {predicho} , Real: {real}")