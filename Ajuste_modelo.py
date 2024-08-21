#ETAPA DE AJUSTE DEL MODELO

#Librerías a utilizar en esta etapa
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar los datos preprocesados
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()  # Convertir DataFrame a array
y_test = pd.read_csv('y_test.csv').values.ravel()  # Convertir DataFrame a array

# Cargar el LabelEncoder
le = joblib.load('label_encoder.pkl')

# Definir el clasificador
model = RandomForestClassifier(random_state=42)

# Parámetros para el Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Configurar el Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Entrenar el Grid Search
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros encontrados
print("Mejores Parámetros:")
print(grid_search.best_params_)

# Evaluar el modelo con los mejores parámetros
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluar el mejor modelo
print("Reporte de Clasificación con Mejores Parámetros:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matriz de Confusión con Mejores Parámetros')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()