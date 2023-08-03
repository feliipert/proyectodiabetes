import streamlit as st
# Comando para subir los cambios streamlit run Home.py --server.runOnSave True


st.set_page_config(
    page_title= "Clasificador de Diabetes Positivo o Negativo",
)

st.write("# Bienvemido al Clasificador de Diabetes ")

st.sidebar.success("Select a demo above.")
st.markdown(
    """
Tabla de Características de la Base de Datos para la Predicción de Diabetes:

| Característica         | Descripción                                                                                                    |
|------------------------|---------------------------------------------------------------------------------------------------------------|
| Gender                 | Género del paciente (Male/Female).                                                                           |
| Age                    | Edad del paciente en años.                                                                                   |
| Hypertension           | Indicador binario de si el paciente tiene hipertensión (0: No, 1: Sí).                                      |
| Heart Disease          | Indicador binario de si el paciente tiene alguna enfermedad cardíaca (0: No, 1: Sí).                       |
| Smoking History        | Historial de tabaquismo del paciente (Never Smoked / Former Smoker / Current Smoker).                      |
| BMI (Body Mass Index)  | Índice de masa corporal del paciente.                                                                       |
| HbA1c Level            | Nivel de hemoglobina A1c en la sangre del paciente (medida del control del azúcar en la sangre).            |
| Blood Glucose Level    | Nivel de glucosa en la sangre del paciente.                                                                  |
| Diabetes               | Variable objetivo: Indicador binario de si el paciente tiene diabetes (0: No, 1: Sí).                       |



Utilidad de la Base de Datos:

La base de datos contiene información médica y características personales de los pacientes, con el propósito de predecir la presencia o ausencia de diabetes. El objetivo es utilizar estas características para entrenar un modelo de aprendizaje automático que pueda realizar predicciones precisas y tempranas sobre la diabetes. La capacidad de predecir la diabetes puede ayudar a los profesionales de la salud a identificar a los pacientes con mayor riesgo de desarrollar esta enfermedad, lo que permitiría una intervención temprana y un seguimiento adecuado para mejorar los resultados de los pacientes. El modelo resultante de esta base de datos puede ser una herramienta valiosa en la detección y prevención de la diabetes, y en la toma de decisiones clínicas. 

| Gender | Age | Hypertension | Heart Disease | Smoking History | BMI (Body Mass Index) | HbA1c Level | Blood Glucose Level | Diabetes |
|--------|-----|--------------|---------------|-----------------|----------------------|-------------|--------------------|----------|
| Male   | 45  | 1            | 0             | Former Smoker   | 25.3                 | 6.2         | 110                | 1        |
| Female | 62  | 0            | 1             | Never Smoked    | 29.8                 | 7.8         | 145                | 0        |
| Female | 38  | 0            | 0             | Current Smoker  | 22.1                 | 5.9         | 95                 | 0        |
| Male   | 55  | 1            | 1             | Former Smoker   | 27.5                 | 7.2         | 120                | 1        |
| Female | 71  | 1            | 0             | Never Smoked    | 31.1                 | 6.5         | 135                | 1        |

Esta tabla representa una pequeña muestra de la base de datos, donde cada fila corresponde a un paciente y cada columna representa una característica específica de ese paciente. Por ejemplo, en la primera fila, el paciente es de género masculino, tiene 45 años, tiene hipertensión (Hypertension=1), no tiene enfermedad cardíaca (Heart Disease=0), tiene un historial de tabaquismo anterior (Smoking History=Former Smoker), su índice de masa corporal es 25.3, su nivel de hemoglobina A1c es 6.2, su nivel de glucosa en la sangre es 110, y tiene diabetes (Diabetes=1).


    """
)
