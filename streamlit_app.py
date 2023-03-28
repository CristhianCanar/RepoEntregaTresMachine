import numpy as np
import streamlit as st
import pickle

# Loading th saved model
#loaded_model = pickle.load(open('C:\\Users\\crist\\Documents\\UNIVERSIDAD\\Semestre IX\\MachineLearning\\consignas\\RepoEntregaTres\\trained_model.sav', 'rb'))

def salaryPrediction(inputData):
    # Parse to numpy array
    inputDataAsNumpyArray = np.asarray(inputData)

    # Reshape the array as we are predicting for one instance(REVISAR BIEN ESTE PUNTO)
    inputDataReshaped = inputDataAsNumpyArray.reshape(1, -1)
    
    # Calculate prediction
    #prediction = loaded_model.predict(inputDataReshaped)
    #print(prediction)
    prediction = [0]
    if(prediction[0] == 0):
        return 'La persona ganará mas de 50mil dolares al año'
    else:
        return 'La persona no ganará 50mil dolares al año :('

def main():
    # Giving a title
    st.title('Prediccion para salario de $50K al año')

    #Getting the input data from the user
    inputUno    = st.text_input('Input uno')
    inputDos    = st.text_input('Input dos')
    inputTres   = st.text_input('Input tres')
    options     = st.multiselect('What are your favorite colors',
                                ['Green', 'Yellow', 'Red', 'Blue'],
                                ['Yellow', 'Red'])
    st.write('You selected:', options)
    # Code for prediction
    prediction = ''

    #Creating button for prediction
    if st.button('Comprobar si ganaré mas de 50K al año'):
        prediction = salaryPrediction([inputUno, inputDos, inputTres])

    st.success(prediction)


if __name__ == '__main__':
    main()
