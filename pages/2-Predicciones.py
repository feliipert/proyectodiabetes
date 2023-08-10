import pickle

import pandas as pd
import requests
import streamlit as st

with open ('models/modelo_RF.pkl' , 'rb') as gb: #rb lectura
    modelo = pickle.load(gb)

st.subheader(" 👨‍💻 Modelo de Machine Learning 👩‍💻️ ")
st.subheader(" Predicción Diabetes Positivo o Negativo 🔬 ")
st.subheader("Caracteristicas de entrada")
st.write("A continuación, ingrese los valores de las caracteristicas de entrada")

def user_input_parameters():
    features = ['Géneros 0: Hombre, 1: Mujer, 2: Otro', 'Edad', 'Hipertensión (0: No, 1: Sí). ',
                'Enfermedad cardíaca (0: No, 1: Sí)',
                ' Tabaquismo (0: No info, 1: Nunca, 2: Fumó, 3: Fuma,4: a veces, 5: Alguna vez:  .',
                'Índice de masa corporal (Valor entre 10 y 95.7) ',
                'Nivel de hemoglobina A1c en la sangre (Valor entre 3.5 y 9)',
                'Nivel de glucosa en la sangre (Valor entre 80 y 300)']
    inputs = {}
    for i, feature in enumerate(features):
        row, col = i // 2, i % 2
        with st.container():
            if i % 2 == 0:
                cols = st.columns(2)
            inputs[feature] = cols[col].text_input(feature)
    data_features = {
        'gender': inputs[features[0]],
        'age': inputs[features[1]],
        'hypertension': inputs[features[2]],
        'heart_disease': inputs[features[3]],
        'smoking_history': inputs[features[4]],
        'bmi': inputs[features[5]],
        'HbA1c_level': inputs[features[6]],
        'blood_glucose_level': inputs[features[7]]
    }
    features_df = pd.DataFrame(data_features, index=[0])
    return features_df

def solicitud_API(muestra_df: pd.DataFrame):
    #url de la API
    #url = 'http://appwebdiabetes.azurewebsites.net/predict'
    url = 'https://appwebdiabetes.azurewebsites.net/predict'
    #url = 'http://127.0.0.1:8000/predict'

    #Datos de entrada
    data = {"data": muestra_df.to_dict(orient='records')}
    response = requests.post(url, json=data)
    #Verificar el codigo de estado de la respuesta
    if response.status_code == 200:
        #Obtener la respuesta en formato JSON
        result = response.json()
        #Obtener la predicción
        prediction = result["prediction"]
        print("Predicion:", prediction)
        return prediction
    else:
        print("Error en la solicitud:", response.status_code)
        return None

df = user_input_parameters()
# Crear un nuevo DataFrame con una fila adicional 'Valor'
df = df.T.reset_index()
df.columns = ['Caracteristica', 'Valor']
df = df.set_index('Caracteristica').T
st.table(df)

# Crear dos botones 'PREDECIR'
predict_button, clear_button = st.columns(2)
predict_clicked = predict_button.button('PREDECIR')
clear_clicked = clear_button.button('Limpiar')  # Agregamos el botón 'Limpiar'

prediction = ''

if predict_clicked:
    # Validar que todos los campos contengan números
    for value in df.values.flatten():
        if not value or not value.isdigit():
            st.warning("Por Favor, complete todos los datos")
            break
    else:
        # Hacer la predicción solo si todos los campos contienen números
        prediction = solicitud_API(df)

if clear_clicked:
    # Limpiar la predicción si se hace clic en el botón 'Limpiar'
    prediction = ''

# Mostrar la predicción
if prediction == 0:
    st.success("Resultado Negativo")
elif prediction == 1:
    st.success("Resultado Positivo")
