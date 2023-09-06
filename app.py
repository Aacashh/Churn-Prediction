import streamlit as st
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np

labelencoder = LabelEncoder()

model = torch.load('churn-nn-model')
model.eval()

st.title("Customer Churn Prediction")
st.write("## Predict if a customer will churn using machine learning and deep learning.")
st.write("Enter the customer details in the sidebar and click on the button below to predict.")

st.sidebar.header("User Input Parameters")
age = st.sidebar.slider("Age", 18, 70, 44)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
location = st.sidebar.selectbox("Location", ["Los Angeles", "New York", "Miami"])
subscription_length = st.sidebar.slider("Subscription Length (months)", 1, 24, 12)
monthly_bill = st.sidebar.slider("Monthly Bill ($)", 30.0, 100.0, 65.0)
total_usage = st.sidebar.slider("Total Usage (GB)", 50.0, 500.0, 274.0)

encoded_gender = labelencoder.fit_transform([gender])[0]
encoded_location = labelencoder.fit_transform([location])[0]
input_data = np.array([age, encoded_gender, encoded_location, subscription_length, monthly_bill, total_usage])
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

if st.button("Predict"):
    with st.spinner("Predicting..."):
        with torch.no_grad():
            prediction = model(input_tensor)

    
    st.header("Prediction")
    if prediction.item() > 0.5:
        st.success("The customer is likely to churn.")
    else:
        st.success("The customer is likely to stay.")
