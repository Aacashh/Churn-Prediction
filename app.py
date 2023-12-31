import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle

class DenserNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(DenserNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.layer4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.5)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = torch.relu(self.layer4(x))
        x = self.dropout4(x)
        x = torch.relu(self.layer5(x))
        x = self.layer6(x)
        return x

model = torch.load('churn-nn-model', map_location=torch.device('cpu'))
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

with open('gen_enc.pkl', 'rb') as f:
    gen_enc = pickle.load(f)

with open('loc_enc.pkl', 'rb') as g:
    loc_enc = pickle.load(g)
# gen_enc.classes_ = np.load('gen_enc.npy', allow_pickle=True).item()
# loc_enc.classes_ = np.load('loc_enc.npy', allow_pickle=True).item()
print(gender + " " + location)
encoded_gender = gen_enc.transform([gender])[0]
encoded_location = loc_enc.transform([location])[0]

bill_to_usage_ratio = monthly_bill / total_usage if total_usage != 0 else 0
age_x_subscription_length = age * subscription_length

input_data = np.array([age, encoded_gender, encoded_location, subscription_length, monthly_bill, total_usage, bill_to_usage_ratio, age_x_subscription_length])
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

if st.button("Predict"):
    with st.spinner("Predicting..."):
        # To display input tensor in the app
        st.write("Input tensor:", input_tensor)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        # To display probabilities in the app
        probs = torch.nn.functional.softmax(output, dim=1)
        st.write("Probabilities:", probs)
        predicted_class = torch.argmax(probs, dim=1).item()
        st.write("Predicted class:", predicted_class)
    # Display Prediction
    st.header("Prediction")

    if predicted_class == 1:
        st.success("The customer is likely to churn.")
    else:
        st.success("The customer is likely to stay.")
