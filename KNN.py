import numpy as np
import pandas as pd
from collections import Counter

class KNN_Manual:
    def __init__(self, k=3):
        self.k = k #Cantidad de vecionos a tomar en cuenta

    def fit(self, X, y):
        self.X_train = np.array(X) #Recordemos que KNN no construye un modelo generalizado , ni llega a aprender parámetros
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X) #Matriz de nuevas muestras a predecir
        predicciones = [self._predict_single(x) for x in X]
        return np.array(predicciones)

    def _predict_single(self, x):
        distancias = np.sqrt(np.sum((self.X_train - x)**2, axis=1)) #Calcular distncia Euclidiana
        
        k_indices = np.argsort(distancias)[:self.k] #Se obtienen los indices de los k vecinos
        
        k_etiquetas = [self.y_train[i] for i in k_indices] #se obtienen las etiquetas
        
        voto_mayoritario = Counter(k_etiquetas).most_common(1) #Se vota a la clase mayoritaria entre los k vecinos
        return voto_mayoritario[0][0]

