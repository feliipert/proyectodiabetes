import matplotlib
import streamlit as st
import pandas as pd
import seaborn as sns
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
    Carga de los datos desde el DATA SET ORIGINAL 
    """
)
st.dataframe(d.head(50))

st.subheader(" Analisis exploratorio de datos (Exploratory data analysis -EDA ")
#FRECUENCIA POR CLASE
#Realizamos un conte de las clases que hay en la base de datos (Cuantos Positivos y Cuantos Negativos)
st.write("Frecuencia por Clase")
ax = d['diabetes'].value_counts()
st.dataframe(ax)

valores = d['diabetes'].value_counts().tolist()

etiquetas = d['diabetes'].unique()

# Crear el gráfico de barras utilizando Matplotlib
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(etiquetas, valores)

# Agregar título y etiquetas a los ejes
ax.set_title('Gráfico de Barras')
ax.set_xlabel('Etiquetas')
ax.set_ylabel('Valores')

# Mostrar el gráfico de barras en Streamlit utilizando 'st.pyplot()'
st.pyplot(fig)

# Cuenta de datos en las features, se puede verificar que no haya features con valores nulos
st.write("Hacemos un conteo de los datos en la featrues, para verificar que hoy valores nulos")
st.write(d.groupby("diabetes").count())

# Reemplazar los valores "Masculino" por 0 , "Femenino" por 1 y "Other" por 2 en la columna "genero"
# Verificar si todos los valores en la columna 'gender' son numéricos o representan números enteros
if d['gender'].apply(lambda x: isinstance(x, (int, float))).all():
    d['gender'] = d['gender'].astype('int64')
else:
    st.write("No todos los valores en la columna 'gender' son numéricos o representan números enteros.")

# Mostrar el DataFrame resultante en Streamlit
st.title("Procesamiento de Datos")
st.write("DataFrame resultante:")
st.write(d)



# Reemplazar los valores smoking_history "No info" por 0 , "Never" por 1 y "former" por 2, "current" por 3, "not current" por 4, "ever" por 5 en la columna "genero"
d['smoking_history'] = d['smoking_history'].replace(['No Info', 'never', 'former', 'current', 'not current', 'ever'], [0, 1, 2, 3, 4, 5])

if d['smoking_history'].apply(lambda x: isinstance(x, (int, float))).all():
    d['smoking_history'] = d['smoking_history'].astype('int64')
else:
    print("No todos los valores en la columna son numéricos o representan números enteros.")


st.write("Mostramos las estadísticas básicas en las "
         "features: media, desviación estandar, valor mínimo, cuartiles, y valor máximo")
st.write(d.describe())

st.write("Histograma de cada feature, para ver las distribuciones en cada feature y detectar alguna anómala o con pocos datos fuera de rango o incluso features nulas")
# Iterar a través de las columnas del DataFrame y mostrar los histogramas
for column in d.columns:
    fig, ax = plt.subplots()
    ax.hist(d[column], bins=20)
    ax.set_title(f'Histograma de {column}')
    st.pyplot(fig)
    matplotlib.pyplot.close()

#correlation_matrix = d.corr()

# Mostrar la matriz de correlación en Streamlit
#st.title("Matriz de Correlación")

# Utilizar Seaborn para crear un mapa de calor de la matriz de correlación
#fig, ax = plt.subplots()
#sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
#st.pyplot(fig)

# Pairplot
#sns.pairplot(d, hue="diabetes")

# Construyendo diagramas de cajon para representar la distribucion del conjunto de datos numéricos a través de sus cuartiles
#nrows=2
#ncols=3

#fig = plt.figure(figsize=(22,15))
#fig.subplots_adjust(hspace=0.2, wspace=0.1)


