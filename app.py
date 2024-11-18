import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

st.write('HEllo babe')

#load the trained model
model=tf.keras.models.load_model('model.h5')

#load the encoders and Scaler
with open ('label_encoder_gender.pkl','rb')as file:
    label_encoder_gender=pickle.load(file)

with open('OHE_geography_column.pkl','rb') as file:
    OHE_geography_column=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# #Example input data
# input_data={'CreditScore':600,
#             'Geography':'France',
#             'Gender':'Male',
#             'Age':40,
#             'Tenure':3,
#             'Balance':60000,
#             'NumOfProducts':2,
#             'HasCrCard':1,
#             'IsActiveMember':1,
#             'EstimatedSalary':50000
#             }

#streamlit app
st.title("Customer churn Prediction")

#User Input
geography=st.selectbox('Geography',['France','Germany','Spain'])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has credit Card',[0,1])
is_active_member=st.select_slider('Is active member',[0,1])

#Prepare the input data
#Example input data
input_data=pd.DataFrame({'CreditScore':[credit_score],
            'Gender':[label_encoder_gender.transform([gender])[0]],
            'Age':[age],
            'Tenure':[tenure],
            'Balance':[balance],
            'NumOfProducts':[num_of_products],
            'HasCrCard':[has_cr_card],
            'IsActiveMember':[is_active_member],
            'EstimatedSalary':[estimated_salary]
            })

#OHE Encode geography
ohe_geography=OHE_geography_column.transform([[geography]]).toarray()
ohe_geography_df=pd.DataFrame(ohe_geography,columns=OHE_geography_column.get_feature_names_out(['Geography']))

#Combine one-hot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),ohe_geography_df],axis=1)

#scale the input
input_data_scaled=scaler.transform(input_data)

#predict churn
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

if prediction_proba > 0.5:
    st.write('The customer is likely to churn')

else:
    st.write('The customer is not likely to churn')

st.write(f"Churn Probability is {prediction_proba}")