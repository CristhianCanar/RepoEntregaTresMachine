import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Configura el dispositivo de entrada/salida para cargar los archivos del modelo
#load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')

# Carga el modelo guardado usando tf.saved_model.load()
loaded_model = tf.keras.models.load_model('train/')
# Loading th saved model

def salaryPrediction(inputData, listado_num):
    entradas_web = pd.read_csv("df_pruebas.csv",
                            delimiter= ",", decimal= ".")
    entradas_web_cat = pd.read_csv("entradas_df_norm_pruebas.csv",
                        delimiter= ",", decimal= ".")
    df_cat = entradas_web_cat.iloc[:,0:21]
    nuevo_df_num = entradas_web.loc[:,['age','hours_per_week']]
    input_array_num = np.asarray(listado_num)
    reshape_num = input_array_num.reshape(1,-1)

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(nuevo_df_num)
    df_norm_input = min_max_scaler.transform(reshape_num)
    df_norm_input = pd.DataFrame(df_norm_input, columns=nuevo_df_num.columns)

    input_array_cat = np.asarray(inputData)
    reshape_cat = input_array_cat.reshape(1,-1)
    df_norm_cat = pd.DataFrame(reshape_cat, columns=df_cat.columns)

    df_final = pd.concat([df_norm_cat,df_norm_input], axis=1)

    # Calculate prediction
    prediction = loaded_model.predict(df_final)
    print(prediction)
    print(np.round(prediction[0]))
    if(np.round(prediction[0]) == 0): 
        return 'La persona ganará mas de 50mil dolares al año 🤩'
    else:
        return 'La persona no ganará 50mil dolares al año 😔'

def validarEducacion(education):
    education_tuple = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    if(education=='educación_10º grado'):
        education_tuple = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    if(education=='educación_11º grado'):
        education_tuple = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    if(education=='educación_12º grado'):
        education_tuple = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
    if(education=='educación_1º-4º grado'):
        education_tuple = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
    if(education=='educación_5º-6º grado'):
        education_tuple = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
    if(education=='educación_7º-8º grado'):
        education_tuple = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    if(education=='educación_9º grado'):
        education_tuple = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    if(education=='educación_Asociado en artes y ciencias'):
        education_tuple = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
    if(education=='educación_Asociado en estudios vocacionales'):
        education_tuple = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    if(education=='educación_Licenciatura/Profesional'):
        education_tuple = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    if(education=='educación_Doctorado'):
        education_tuple = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    if(education=='educación_Diploma de escuela secundaria'):
        education_tuple = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
    if(education=='educación_Maestría'):
        education_tuple = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
    if(education=='educación_Educación preescolar'):
        education_tuple = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
    if(education=='educación_Escuela profesional'):
        education_tuple = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
    if(education=='educación_Algo de universidad'):
        education_tuple = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    return education_tuple


def validarRaza(raza):
    raza_tuple = [1,0,0,0,0]
    if(raza=='raza_Americano indio/esquimal'):
        raza_tuple = [1,0,0,0,0]
    if(raza=='raza_Asian/Pacific Islander'):
        raza_tuple = [0,1,0,0,0]
    if(raza=='raza_Negra'):
        raza_tuple = [0,0,1,0,0]
    if(raza=='raza_Otra'):
        raza_tuple = [0,0,0,1,0]
    if(raza=='raza_Blanca'):
        raza_tuple = [0,0,0,0,1]

    return raza_tuple

def main():
    # Titulo
    st.title('Predicción para salario de $50K al año')

    #Getting the input data from the user
    age                = st.number_input('Edad')
    hours              = st.number_input('Horas trabajadas por semana')
    select_education   = st.selectbox('Selecciona tu nivel de educación alcanzado: ',
                                        ('educación_10º grado',
                                        'educación_11º grado',
                                        'educación_12º grado',
                                        'educación_1º-4º grado',
                                        'educación_5º-6º grado',
                                        'educación_7º-8º grado',
                                        'educación_9º grado',
                                        'educación_Asociado en artes y ciencias',
                                        'educación_Asociado en estudios vocacionales',
                                        'educación_Licenciatura/Profesional',
                                        'educación_Doctorado',
                                        'educación_Diploma de escuela secundaria',
                                        'educación_Maestría',
                                        'educación_Educación preescolar',
                                        'educación_Escuela profesional',
                                        'educación_Algo de universidad',
                                    ),)
            
    select_raza   = st.selectbox('Selecciona tu raza: ',
                            ('raza_Americano indio/esquimal',
                            'raza_Asian/Pacific Islander',
                            'raza_Negra',
                            'raza_Otra',
                            'raza_Blanca'
                            ),)
    tupla_num = [age, hours]
    nivel_educacion = validarEducacion(select_education)
    raza = validarRaza(select_raza)

    entradas_tupla = nivel_educacion+raza

    #Creating button for prediction
    prediction = ''
    if st.button('Comprobar si ganaré mas de 50K al año XD'):
        prediction = salaryPrediction(entradas_tupla, tupla_num)

    st.success(prediction)


if __name__ == '__main__':
    main()
