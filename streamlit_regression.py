import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load trained model
model = tf.keras.models.load_model('regression_model.h5')

# load the encoders and scaler
with open('onehot_encoder_geo_reg.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender_reg.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler_reg.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title('Estimated Salary Prediction')

# user inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, value=38)
balance = st.number_input('Balance', min_value=0.0, value=76000.0)
credit_score = st.number_input('Credit Score', min_value=350, max_value=850, value=650)
exited = st.selectbox('Exited', [0, 1], index=0)
tenure = st.slider('Tenure', 0, 10, value=5)
num_of_products = st.slider('Number of Products', 1, 4, value=1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1], index=1)
is_active_member = st.selectbox('Is Active Member', [0, 1], index=1)

# prepare the input data (same column order as training)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# one hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# combine one hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict salary
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.write(f'Predicted Estimated Salary: ${predicted_salary:,.2f}')
