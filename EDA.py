#ANÁLISIS EXPLOTARIO DE DATOS

# Se importan las librerías a utlizar en esta etapa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Cargar los datos limpios
X = pd.read_csv('features.csv')
y = pd.read_csv('targets.csv')

# Verificar tipos de datos
print("Tipos de datos en las características:")
print(X.dtypes)

# Codificación de variables categóricas
X_encoded = pd.get_dummies(X, columns=[
    'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',  'SCC', 'CALC', 'MTRANS'
], drop_first=True)

#NUEVO
# Imprimir las primeras filas del DataFrame codificado
print("\nPrimeras filas del DataFrame codificado:")
print(X_encoded.head())

#NUEVO
# Verificar las columnas generadas
print("\nColumnas después de codificación one-hot:")
print(X_encoded.columns)

#NUEVO
# Comparar con las columnas originales
print("\nColumnas originales:")
print(X.columns)

#NUEVO
# Ver cómo se codificó una variable específica (por ejemplo, 'Gender')
print("\nVerificación de la codificación de 'Gender':")
print(X[['Gender']].head())
gender_columns = [col for col in X_encoded.columns if 'Gender' in col]
print(X_encoded[gender_columns].head())

# Codificación del objetivo
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

# Unir las características codificadas y objetivos para facilitar el análisis
data = X_encoded.copy()
data['Target'] = y_encoded

# Estadísticas descriptivas
print("\nEstadísticas descriptivas de las características:")
print(X_encoded.describe())

# Revisar valores nulos
print("\nValores nulos en las características:")
print(X_encoded.isnull().sum())

# Distribución de una variable específica 
if 'Age' in X_encoded.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(X_encoded['Age'], kde=True)
    plt.title('Distribución de la Edad')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.show()

# Correlación entre características
plt.figure(figsize=(12, 8))
correlation_matrix = X_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.show()

# Distribución del objetivo
plt.figure(figsize=(10, 6))
sns.histplot(data['Target'], kde=True)
plt.title('Distribución del Objetivo')
plt.xlabel('Objetivo')
plt.ylabel('Frecuencia')
plt.show()

# Relación entre una característica y el objetivo (por ejemplo, 'Age' vs 'Target')
if 'Age' in X_encoded.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Target', data=data)
    plt.title('Relación entre Edad y Objetivo')
    plt.xlabel('Edad')
    plt.ylabel('Objetivo')
    plt.show()
