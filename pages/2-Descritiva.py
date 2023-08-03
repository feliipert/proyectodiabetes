import streamlit as st
import pandas as pd
from urllib.error import URLError
import matplotlib.pyplot as plt

#Importamos los datasets que vamos autilizar

folder = 'data'
archivo_data ='diabetes_prediction_dataset.csv'
data = pd.read_csv('data' + '/' + archivo_data, sep=',')

#Definimos las clases que vamos a utilizar y reemplazmos su valor


#Creamos una copia de la data para reemplazar de la columna
d=data.copy()
# Reemplazar los valores 0  por Positivo , y 1 por Negativo
d['diabetes'] = d['diabetes'].replace([0, 1], ['Positivo', 'Negativo'])

st.set_page_config(page_title="Conjunto de datos")
st.markdown("#Conjunto de datos")
st.sidebar.header("Conjunto de datos")#Como se va a llamar el data Frame
st.write(
    """
    Vamos a visualizar los datos que tiene la base de datos
    """
)
st.dataframe(d.head(50))

#FRECUENCIA POR CLASE
#Realizamos un conte de las clases que hay en la base de datos (Cuantos Positivos y Cuantos Negativos)
ax = d['diabetes'].value_counts()
st.dataframe(ax)

valores = d['diabetes'].value_counts().tolist()

etiquetas = d['diabetes'].unique()

# Crear el gráfico de pastel utilizando Matplotlib
fig, ax = plt.subplots(figsize=(2,2))
ax.pie(valores, labels=etiquetas, autopct='%1.1f%%')
ax.axis('equal')  # Esto ajusta el gráfico para que tenga forma de círculo

# Mostrar el gráfico de pastel en Streamlit utilizando 'st.pyplot()'
st.pyplot(fig)