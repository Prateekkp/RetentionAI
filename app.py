import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Get the absolute path to the 'models' directory
models_dir = os.path.join(os.path.dirname(__file__), "models")

# Load the trained model, scaler, and label encoders
with open(os.path.join(models_dir, "churn_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(models_dir, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)


# Streamlit UI
st.set_page_config(page_title="RetentionAI", page_icon="📊", layout="centered")

st.title("📊 RetentionAI")
st.markdown("### Predict whether a customer is likely to **stay or leave** based on their service details.")

st.markdown("---")

# Get user input
st.subheader("🔍 Enter Customer Details:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
    partner = st.radio("Has Partner?", ["No", "Yes"])
    dependents = st.radio("Has Dependents?", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

    phone_service = st.radio("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check"],  # Only keep known category
    index=0)

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

# Convert input to DataFrame
user_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],  # Convert Yes/No to 1/0
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Preprocess input data
def preprocess_input(data):
    # Encode categorical variables
    for col in label_encoders:
        if col in data:
            data[col] = label_encoders[col].transform(data[col])

    # Scale numeric features
    data_scaled = scaler.transform(data)

    return data_scaled

st.markdown("---")

# Predict Button
if st.button("🔮 Predict Churn"):
    # Preprocess input
    input_data = preprocess_input(user_data)

    # Predict
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of churn

    # Display result
    st.markdown("## 🏆 Prediction Result:")
    
    if prediction == 1:
        st.error(f"⚠️ **This customer is likely to churn!** (Probability: {prediction_proba:.2%})")
    else:
        st.success(f"✅ **This customer is not likely to churn.** (Probability: {prediction_proba:.2%})")

st.markdown("📌 _Developed by **Prateek Kumar Prasad** 🚀_")
