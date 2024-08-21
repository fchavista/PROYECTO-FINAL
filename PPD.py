#ETAPA DE PREPROCESAMIENTO DE DATOS

#Librerías a utilizar en esta etapa
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar los datos limpios
X = pd.read_csv('features.csv')
y = pd.read_csv('targets.csv')

# Codificación de variables categóricas
X_encoded = pd.get_dummies(X, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',  'SCC', 'CALC', 'MTRANS'
], drop_first=True)

# Codificación del objetivo
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# División del dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Guardar los datos preprocesados
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

# Guardar el LabelEncoder para usarlo en el modelado
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"Datos preprocesados guardados con éxito.")