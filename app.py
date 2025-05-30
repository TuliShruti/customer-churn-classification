
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load model and preprocessing objects
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Customer Churn Prediction')

# User input
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 40)
tenure = st.slider('Tenure', 0, 10, 3)
balance = st.number_input('Balance', min_value=0.0, value=60000.0)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1], index=1)
is_active_member = st.selectbox('Is Active Member', [0, 1], index=1)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

if st.button('Predict'):
    # Prepare input as DataFrame
    input_dict = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
    input_df = pd.DataFrame([input_dict])

    # Encode categorical variables
    input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

    # Reorder columns to match training
    expected_cols = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = float(prediction[0][0])

    st.write(f'Churn Probability: {prediction_proba:.2f}')
    if prediction_proba > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
