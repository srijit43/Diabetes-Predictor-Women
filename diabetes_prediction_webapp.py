# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:14:57 2023

@author: sriji
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/sriji/ML_datasets/machine_learning_model_deployment/diabetes_prediction_model.sav','rb'))

#Creating a function for prediction

def diabetes_predictor(input_data):
    #input_data = (5,166,72,19,175,25.8,0.587,51)
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    #std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    
    #giving a title for our web app
    st.title('Diabetes Predictor Web App')
    
    #Getting input data from the user Pregnancies
     #Glucose
     #BloodPressure
     #SkinThickness
     #Insulin
     #BMI
     #DiabetesPedigreeFunction
     #Age
     #Outcome
     
    Pregnancies = st.text_input('Number of pregnancies:')
    Glucose = st.text_input('BLood Glucose level')
    BloodPressure = st.text_input('Blood Pressure level')
    SkinThickness = st.text_input('Thickness of skin')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI tested')
    DiabetesPedigreeFunction = st.text_input('DBP')
    Age = st.text_input('Age of patient')
    
    #Code for prediction
    
    diagnosis = ''
    
    #creating a button for prediction
    
    if st.button('Diabetes test result'):
        diagnosis = diabetes_predictor([Pregnancies, Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()


    