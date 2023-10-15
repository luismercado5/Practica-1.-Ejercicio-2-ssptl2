# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:30:48 2023

@author: luis mercado
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# Lectura del archivo de datos
datos = pd.read_csv('spheres1d10.csv', header=None)

# Extracción de las entradas y las salidas de los datos
entradas = datos.iloc[:, :-1].values
salidas = datos.iloc[:, -1].values

# Definición de parámetros de particionamiento
num_particiones = 5  # Número de particiones
porcentaje_entrenamiento = 0.8  # Porcentaje de patrones de entrenamiento
porcentaje_prueba = 1 - porcentaje_entrenamiento  # Porcentaje de patrones de prueba

# Creación de las particiones de entrenamiento y prueba
particiones_entrenamiento = []
particiones_prueba = []

for _ in range(num_particiones):
    entradas_entrenamiento, entradas_prueba, salidas_entrenamiento, salidas_prueba = train_test_split(
        entradas, salidas, train_size=porcentaje_entrenamiento, test_size=porcentaje_prueba
    )
    particiones_entrenamiento.append((entradas_entrenamiento, salidas_entrenamiento))
    particiones_prueba.append((entradas_prueba, salidas_prueba))

# Creación y entrenamiento del perceptrón para cada partición
for i in range(num_particiones):
    entradas_entrenamiento, salidas_entrenamiento = particiones_entrenamiento[i]
    perceptron = Perceptron()
    perceptron.fit(entradas_entrenamiento, salidas_entrenamiento)
    # Realiza las operaciones que desees con el perceptrón entrenado en cada partición
    # ...

    # Ejemplo: Mostrar el porcentaje de acierto en la partición de prueba
    entradas_prueba, salidas_prueba = particiones_prueba[i]
    salidas_predichas = perceptron.predict(entradas_prueba)
    porcentaje_acierto = (salidas_prueba == salidas_predichas).mean()
    print(f"Partición {i+1}: Porcentaje de acierto en la prueba: {porcentaje_acierto * 100}%")
