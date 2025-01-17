import streamlit as st
import pickle
import numpy as np


df=pickle.load(open('dfs.pkl','rb'))
pipe=pickle.load(open('pipes.pkl','rb'))



st.title('Sleep Disorder')
gender=st.selectbox('Gender',df['Gender'].unique())
age=st.slider('Age',8,20,90)
occupation=st.selectbox('Occupation',df['Occupation'].unique())
sleepDuration=st.slider('Sleep Duration',0,15,5)
qualityofsleep=st.slider('SleepQuality',0,10,5)
phsicalactivity=st.number_input('Physical Activity(in min)')
stresslevel=st.slider('Stress Level',0,10,5)
bmicategory=st.selectbox('BMI Category',df['BMI Category'].unique())
heartrate=st.slider('Heart Rate',60,90,75)
dailysteps=st.number_input('Daily Steps')
systolic=st.number_input('Enter Your Systolic Pressure')
dystolic=st.number_input('Enter your Dystlolic Pressure')
if st.button('Predict'):
   query=np.array([gender,age,occupation,sleepDuration,qualityofsleep,phsicalactivity,stresslevel,bmicategory,heartrate,dailysteps,dystolic,systolic])
   query=query.reshape(1,12)
   curr=pipe.predict(query)


   st.write('Result :- ',str(curr[0]))

