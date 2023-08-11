import matplotlib
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Importamos los datasets que vamos autilizar
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

folder = 'data'
archivo_data ='diabetes_prediction_dataset.csv'
data = pd.read_csv('data' + '/' + archivo_data, sep=',')

#Definimos las clases que vamos a utilizar y reemplazmos su valor


#Creamos una copia de la data para reemplazar de la columna
d=data.copy()
# Reemplazar los valores 0  por Positivo , y 1 por Negativo
#d['diabetes'] = d['diabetes'].replace([0, 1], ['Positivo', 'Negativo'])

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

# Crear un botón para ejecutar el procesamiento de datos
st.write("Ejecutamos los Botones para cambiar los datos de tipo Object a datos de Tipo Int")
if st.button("Ejecutar Para seguir"):
    # Reemplazar los valores en la columna 'smoking_history'
    d['smoking_history'] = d['smoking_history'].replace(['No Info', 'never', 'former', 'current', 'not current', 'ever'], [0, 1, 2, 3, 4, 5])
    d['gender'] = d['gender'].replace(['Male', 'Female', 'Other'], [0, 1, 2])

    # Verificar y convertir la columna 'smoking_history'
    if d['smoking_history'].apply(lambda x: isinstance(x, (int, float))).all():
        d['smoking_history'] = d['smoking_history'].astype('int64')
    else:
        st.write("No todos los valores en la columna 'smoking_history' son numéricos o representan números enteros.")
    if d['gender'].apply(lambda x: isinstance(x, (int, float))).all():
        d['gender'] = d['gender'].astype('int64')
    else:
        st.write("No todos los valores en la columna 'gender' son numéricos o representan números enteros.")

    # Mostrar el DataFrame resultante en Streamlit
    st.title("Procesamiento de Datos")
    st.write("DataFrame resultante:")
    st.write(d)

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

    correlation_matrix = d.corr()

    # Mostrar la matriz de correlación en Streamlit
    st.title("Matriz de Correlación")

    # Utilizar Seaborn para crear un mapa de calor de la matriz de correlación
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    st.pyplot(fig)

    # Mostrar el pairplot en Streamlit
    st.title("Pairplot")
    st.write("Pairplot de características")

    # Crear y mostrar el pairplot utilizando Seaborn
    #pairplot = sns.pairplot(d, hue="diabetes")
    #st.pyplot(pairplot)

    st.title("Proceso de división de datos de entrenamiento y testing")
    # Obtener las features
    features = d.drop(['diabetes'], axis=1)
    st.write(features.head())

    # Obtener los labels
    st.write("Obtenemos las Label")
    labels = d['diabetes']
    st.write(labels.head())
    st.write(features.shape,labels.shape)

    # Separación de la data, con un 20% para testing, y 80% para entrenamiento
    st.write("Separación de la data, con un 20% para testing, y 80% para entrenamiento")
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.20, random_state=1, stratify=labels)
    st.write(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Verificacion de la cantidad de datos para entrenamiento y para testing
    st.write("Verificacion de la cantidad de datos para entrenamiento y para testing")
    st.write("y_train labels unique:", np.unique(y_train, return_counts=True))
    st.write("y_test labels unique: ", np.unique(y_test, return_counts=True))

    st.title("KNN con preprocesamiento de las Features")
    st.write("Cargamos el modelo KNN sin entrenar, despues entrenamos el modelo y posteriormente "
             "obtenemos las metricas logradas")
    # Cargamos el modelo KNN sin entrenar
    model_KNN = KNeighborsClassifier(n_neighbors=3)
    # Entrenamos el modelo KNN
    model_KNN.fit(X_train, y_train)
    # Obtenemos la métrica lograda
    model_KNN.score(X_test, y_test)
    st.write("model_KNN.score(X_test, y_test)", model_KNN.score(X_test, y_test))


    st.subheader("Realizamos el preprocesamiento con el StandardScaler")
    # Definir preprocesamiento
    standard_scaler = preprocessing.StandardScaler()
    # Preprocesar los datos
    X_train_standard = standard_scaler.fit_transform(X_train)
    X_test_standard = standard_scaler.transform(X_test)
    st.write(X_train_standard[0:1])
    st.write(X_train[0:1].to_numpy())

    st.write("Cargamos el modelo KNN sin entrenar con StandardScaler, despues entrenamos el modelo y posteriormente "
             "obtenemos las metricas logradas")

    # Cargamos el modelo KNN sin entrenar
    model_KNN = KNeighborsClassifier(n_neighbors=3)
    # Entrenamos el modelo KNN
    model_KNN.fit(X_train_standard, y_train)
    # Obtenemos la métrica lograda
    model_KNN.score(X_test_standard, y_test)
    st.write("model_KNN.score(X_test, y_test) con StandardScaler", model_KNN.score(X_test, y_test))


    st.subheader("Realizamos el proceso con MinMaxScaler")
    # Definir preprocesamiento
    min_max_scaler = preprocessing.MinMaxScaler()
    # Preprocesar los datos
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.transform(X_test)

    st.write("Cargamos nuevamente el modelo KNN sin entrenar, despues entrenamos el modelo y posteriormente obtenemos las metricas logradas")
    # Cargamos el modelo KNN sin entrenar
    model_KNN = KNeighborsClassifier(n_neighbors=3)
    # Entrenamos el modelo KNN
    model_KNN.fit(X_train_minmax, y_train)
    # Obtenemos la métrica lograda
    model_KNN.score(X_test_minmax, y_test)
    st.write("model_KNN.score(X_test, y_test) MinMax Scaler", model_KNN.score(X_test, y_test))

    st.subheader("Balance de Clases")
    st.write("Realizamos un balance de clases hacia la clase mayor con RandomOverSampler")
    # Balance de clases hacia la clase mayor
    sampler = RandomOverSampler(random_state=1)
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)

    st.write("y_test  original: ", np.unique(y_test, return_counts=True))
    st.write("y_train original: ", np.unique(y_train, return_counts=True))
    st.write("y_train balanced: ", np.unique(y_train_balanced, return_counts=True))

