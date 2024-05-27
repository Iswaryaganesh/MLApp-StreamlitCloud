# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle as pkl
import streamlit as st

loaded_model = pkl.load(open('trained_model.sav','rb'))

def diabetes_prediction(input_data):
    #convert to numpy array
    np_array = np.asarray(input_data)
    
    #reshape array as we are predicting for 1 instance
    reshaped_array = np_array.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_array)
    if prediction[0] == 1:
      return 'The person is diabetic'
    else:
      return 'The person is not diabetic'
  
def main():
    st.title('Diabetic prediction web app')
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    diagnosis = ''
    btn = st.button('Test')
    if(btn):
        diagnosis = diabetes_prediction((Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age))
    if(diagnosis == 'The person is diabetic'):
        st.success(diagnosis)
    else:
        st.error(diagnosis)
        
if __name__ == '__main__':
    main()