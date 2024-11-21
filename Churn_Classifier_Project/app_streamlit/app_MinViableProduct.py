# app.py created in ./Churn_Classification_Project/app_streamlit

import streamlit as st
import pandas as pd 
import pickle


# Title of the app
st.title('Customer Churn Predictor')

st.markdown('Enter customer information to predict churn')


# Function to load the model and make predictions
def predict_churn(age, gender, tenure, usage_frequency, support_calls, payment_delay, contract_length,
                  total_spend, last_interaction, subscription_type):
    # Load the saved model
    with open('../models/FINAL_trained_model_02_DecisionTreeClass.pkl', 'rb') as f:
        model = pickle.load(f)

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'tenure': [tenure],
        'usage_frequency': [usage_frequency],
        'support_calls': [support_calls],
        'payment_delay': [payment_delay],
        'subscription_type': [subscription_type],
        'contract_length': [contract_length],
        'total_spend': [total_spend],
        'last_interaction': [last_interaction]
    })

    # Predict churn
    prediction = model.predict(input_data)[0]  # Assuming model outputs 0 or 1 for churn
    return prediction


# Main function for the input form and prediction button
def user_input_features():
    age = st.number_input("Age", min_value=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    tenure = st.slider('Tenure', min_value=1, max_value=60)
    usage_frequency = st.number_input("Usage Frequency", min_value=1)
    support_calls = st.number_input("Total Number of Support Calls", min_value=0)
    payment_delay = st.number_input("Total Payment Delay (in Days)", min_value=0)
    subscription_type = st.selectbox('Subscription Type', ['Basic', 'Standard', 'Premium'])
    contract_length = st.selectbox('Contract Length', ['Monthly', 'Quarterly', 'Annually'])
    total_spend = st.number_input("Total Spend", min_value=1)
    last_interaction = st.number_input("Last interaction (days ago)", min_value=1)

    # Encode categorical variables as needed
    gender_encoded = 1 if gender == "Male" else 0
    contract_length_encoded = {
        'Monthly': 1,
        'Quarterly': 3,
        'Annually': 12
    }[contract_length]

    subscription_type_encoded = {
        'Basic': 1,
        'Standard': 2,
        'Premium': 3
    }[subscription_type]    

    return age, gender_encoded, tenure, usage_frequency, support_calls, payment_delay, \
           subscription_type_encoded, contract_length_encoded, total_spend, last_interaction



# Get user inputs
inputs = user_input_features()

# Predict button
if st.button("Predict Churn"):
    # Make prediction
    prediction = predict_churn(*inputs)
    if prediction == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")


#terminal: streamlit run app.py
