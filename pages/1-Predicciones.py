import streamlit as st
import pandas as pd
import pickle
import sklearn
import requests

def solicitud_API(muestra: list):
    #url de la API
    url = 'https://20.119.16.35/predict'
    #Datos de entrada
    data = {"data": [muestra] }
    response = requests.post(url, json=data)
    #Verificar el codigo de estado de la respuesta
    if response.status_code == 200:
        #Obtener la respuesta en formato JSON
        result = response.json()
        #Obtener la predicción
        prediction = result["prediction"]
        #imprimir la predicción
        return prediction
    else:
        print("Error en la solicitud:", response.status_code)
        return None

with open ('models/modelo_RF.pkl' , 'rb') as gb: #rb lectura
    modelo = pickle.load(gb)

st.subheader(" Machine Larning modelo seleccionado")
st.subheader(" Algoritmo de Machine Learning ")
st.write(" Definición del algoritmo implementado para predecir la Diabetes ")
st.subheader("Caracteristicas de entrada")
features = ['Géneros 0: Hombre, 1: Mujer, 2: Otro', 'Edad', 'Hipertensión (0: No, 1: Sí). ', 'Enfermedad cardíaca (0: No, 1: Sí)', ' Tabaquismo (0: No info, 1: Nunca, 2: Fumó, 3: Fuma,4: a veces, 5: Alguna vez:  .', 'Índice de masa corporal (Valor entre 10 y 95.7) ', 'Nivel de hemoglobina A1c en la sangre (Valor entre 3.5 y 9)', 'Nivel de glucosa en la sangre (Valor entre 80 y 300)']
st.write("A continuación, ingrese los valores de las caracteristicas de entrada")

def user_input_parameters():
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

df = user_input_parameters()
st.subheader("Modelo de Machine Learning")

# Crear un nuevo DataFrame con una fila adicional 'Valor'
df = df.T.reset_index()
df.columns = ['Caracteristica', 'Valor']
df = df.set_index('Caracteristica').T
st.table(df)

#Crear dos botones 'PREDECIR'
predict_button, clear_button = st.columns(2)
predict_clicked = predict_button.button('PREDECIR')
prediction = ''
# Convertir el valor de y_pred_RF a la palabra correspondiente
#resultado = "Positivo" if y_pred_RF == 0 else "Negativo"
if predict_clicked:
    #Validar que todos los campos contengan numeros
    for value in df.values.flatten():
        if not value or not value.isdigit():
            st.warning("Por Favor, complete todos los datos")
            break
        else:
            #prediction = modelo.predict(df)
            prediction = solicitud_API(df.values.flatten().tolist())

    # Crear un diccionario para asociar las predicciones
    prediction_descriptions = {
        0: 'EL RESULTADO ES NEGATIVO',
        1: 'EL RESULTADO ES POSITIVO'
    }
    st.success(prediction_descriptions[prediction])