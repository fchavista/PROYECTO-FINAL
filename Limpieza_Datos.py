#ETAPA DE LIMPIEZA DE DATOS

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos
dataset = fetch_ucirepo(id=544)

# Extraer características y objetivos
X = dataset.data.features
y = dataset.data.targets

# Guardar los datos en archivos CSV 
X.to_csv('features.csv', index=False)
y.to_csv('targets.csv', index=False)

# Mostrar las primeras filas para verificar la carga
print("Primeras filas de las características:")
print(X.head())

print("\nPrimeras filas de los objetivos:")
print(y.head())

# Revisar valores nulos en las características
print("\nValores nulos en las características:")
print(X.isnull().sum())

# Eliminar filas con valores nulos
X = X.dropna()
y = y.loc[X.index]  

# Revición de valores nulos después de eliminación
print("\nValores nulos en las características después de limpieza:")
print(X.isnull().sum())

# Eliminar duplicados
X = X.drop_duplicates()

print("\nPrimeras filas después de eliminar duplicados:")
print(X.head())

# Codificación one-hot para variables categóricas
X = pd.get_dummies(X, drop_first=True)

print("\nPrimeras filas después de la codificación one-hot:")
print(X.head())

#Normalización de Datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertir de nuevo a DataFrame para impresión
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nPrimeras filas después de la normalización:")
print(X_scaled.head())