#GENERACIÓN DEL CUESTIONARIO A NIVEL WEB A TRAVES DE STREAMLIT

#Librerías a utilizar en esta etapa
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar el modelo entrenado y los preprocesadores
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Leer las características del modelo entrenado para conocer el orden correcto de las columnas
X_train_example = pd.read_csv('features.csv')  # Utilizamos 'features.csv' para obtener las columnas correctas
X_train_encoded_example = pd.get_dummies(X_train_example, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
], drop_first=True)

# Función de predicción
def predicción_obesidad(gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans):
    # Crear un DataFrame con la entrada del usuario
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })

    # Codificar las variables categóricas usando pd.get_dummies
    input_data_encoded = pd.get_dummies(input_data, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ], drop_first=True)

    # Hacer que las columnas del DataFrame coincidan con las del modelo
    input_data_encoded = input_data_encoded.reindex(columns=X_train_encoded_example.columns, fill_value=0)

    # Verificar que todos los datos sean numéricos
    try:
        input_data_encoded = input_data_encoded.astype(float)
    except ValueError as e:
        st.error(f"Error en la conversión de datos a numérico: {e}")
        return None

    # Normalizar los datos
    input_data_scaled = scaler.transform(input_data_encoded)

    # Realizar la predicción
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Configuración de Streamlit
st.title('Estimación Nivel de Obesidad')
st.write("Ingrese los siguientes datos para estimar el nivel de obesidad:")

# Campos de entrada
gender = st.number_input('Género (0 para femenino, 1 para masculino):', min_value=0, max_value=1)
age = st.number_input('Edad:', min_value=0, max_value=120)
height = st.number_input('Altura (m):', min_value=0.0, max_value=3.0)
weight = st.number_input('Peso (kg):', min_value=0, max_value=300)
family_history = st.number_input('¿Tiene antecedentes familiares de sobrepeso? (0 para no, 1 para sí):', min_value=0, max_value=1)
favc = st.number_input('¿Consume alimentos altos en calorías frecuentemente? (0 para no, 1 para sí):', min_value=0, max_value=1)
fcvc = st.number_input('¿Come verduras en sus comidas? (0 para no, 1 para sí):', min_value=0, max_value=1)
ncp = st.number_input('Número de comidas principales diarias:', min_value=1, max_value=10)
caec = st.number_input('¿Come algo entre comidas? (0 para no, 1 para alguna veces, 2 para frecuentemente):', min_value=0, max_value=2)
smoke = st.number_input('¿Fuma? (0 para no, 1 para sí):', min_value=0, max_value=1)
ch2o = st.number_input('¿Cuánta agua bebe diariamente? (litros):', min_value=0.0, max_value=10.0)
scc = st.number_input('¿Monitorea las calorías que consume diariamente? (0 para no, 1 para sí):', min_value=0, max_value=1)
faf = st.number_input('Frecuencia de actividad física (días en la semana):', min_value=0, max_value=7)
tue = st.number_input('Tiempo de uso de dispositivos tecnológicos (horas/día):', min_value=0, max_value=24)
calc = st.number_input('Frecuencia de consumo de alcohol (0 para nunca, 1 para raramente, 2 para frecuentemente)', min_value=0, max_value=2)
mtrans = st.number_input('Transporte usado normalmente (0 para transporte_público, 1 para carro, 2 para cicla, 3 para camina):', min_value=0, max_value=3)

# Botón para realizar la predicción
if st.button('Predecir Nivel de Obesidad'):
    try:
        prediction = predicción_obesidad(gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans)
        if prediction is not None:
            st.write(f'Nivel de Obesidad Predicho: {prediction}')
        else:
            st.error("No se pudo realizar la predicción. Verifique los datos ingresados.")
    except Exception as e:
        st.error(f"Se produjo un error: {e}")