from fastapi import FastAPI
import pandas as pd
import pickle

#Cargamos el modelo previamente entrenado
with open ('models/modelo_RF.pkl' , 'rb') as gb: #rb lectura
    modelo = pickle.load(gb)

app = FastAPI()

@app.get('/')
def hello():
    return {'message': 'Hello World'}




