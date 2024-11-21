# app.py created in ./Churn_Classification_Project/app_streamlit

import streamlit as st
import pandas as pd 
import numpy as np
import pickle


# Titles
st.markdown('<h1 style="text-align: left; color: #2c3e50;">Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: left; color: #7f8c8d;">Enter customer data and get predictions</h3>', unsafe_allow_html=True)



def predict_churn(data):

    with open('../models/FINAL_trained_model_02_DecisionTreeClass.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(data)
    return predictions

#file uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file:
    try:
        
        data = pd.read_csv(uploaded_file)

        # column check
        expected_columns = [
            'age', 'gender', 'tenure', 'usage_frequency', 'support_calls', 
            'payment_delay', 'subscription_type', 'contract_length', 
            'total_spend', 'last_interaction'
        ]
        if all(col in data.columns for col in expected_columns):
            # ecoding categorical to numeric
            data['gender'] = data['gender'].map({'Female': 0, 'Male': 1}).fillna(0)
            data['contract_length'] = data['contract_length'].map({
                'Monthly': 1, 'Quarterly': 3, 'Annually': 12
            }).fillna(0)
            data['subscription_type'] = data['subscription_type'].map({
                'Basic': 1, 'Standard': 2, 'Premium': 3
            }).fillna(0)

            # No NaNs
            if data.isnull().values.any():
                st.error("The uploaded file contains missing values. Please clean the data and try again.")
            else:
                
                predictions = predict_churn(data)
                data['Churn_Prediction'] = predictions
                # making prediction easy to interpret for non-technical end user. 
                data['Churn_Prediction'] = np.where(data['Churn_Prediction'] == 1, 'Churn', 'No Churn')

                # print value of predictions
                st.write("Potential Savings:")
                churn_data = data[data['Churn_Prediction']=='Churn']
                churn_total_value = churn_data['total_spend'].sum()*0.9 # 90% precision in testing
                st.write(f'The total value of predicted churn clients, taking into account a 10% margin of error, is £{churn_total_value:.2f}')

                # user can download csv of predictions and see a preview
                csv = data.to_csv(index=False)
                st.write("### Customer Information Preview")
                st.dataframe(data.head(10))  # Show a preview of the dataset

                st.write("### Download Prediction Results")
                st.download_button(label="Download Prediction", data=csv, file_name="churn_predictions.csv", mime="text/csv")



                
        else:
            st.error(f"The uploaded CSV file must contain the following columns: {', '.join(expected_columns)}")
    
    # last potential error message
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")



# page configuration


#sidebar for model information
with st.sidebar:
    st.header("Decision Tree Classifier")
    st.subheader('Development Information')
    st.markdown('Through use of open source technology, a Decision Tree Classifier was trained and tested \
                on over 500,000 different clients. It was favoured over an ensemble model in view of its \
                light weight and speed.')
    st.subheader('Performance')
    st.markdown('This particular classifier was hyperparameterised to optimise recall over precision. It is \
                capable of detecting 99% of all clients that would otherwise go on to churn. With respect to \
                all of those clients identified by the model as "churn", the margin of error is only 10%. ')
    st.subheader('Under the Hood')
    st.markdown('Parameters were fine tuned using Random Search. A balanced class weight makes the model resilient \
                to imbalances, and regularisation techinques allow the model to generalise well with new data. This \
                particular iteration has a max depth of 10, limits leaf samples to 5, and limits splits to 2. \
                The model itself is inherently resilient to outliers.')
    # Add more input fields here


# professional font 
st.markdown(
    """
    <style>
    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    """, 
    unsafe_allow_html=True
)

#custom buttoms and icons 
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #3498db;  # Blue background
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    </style>
    """, unsafe_allow_html=True
)

#footer 
st.markdown(
    """
    <footer style="text-align: center; color: #95a5a6; padding: 20px 0;">
        <p>Created by Nathan Jones | Powered by Streamlit</p>
    </footer>
    """, unsafe_allow_html=True
)

