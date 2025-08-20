import streamlit as st
import pandas as pd
import joblib

# Load trained RF pipeline
@st.cache_resource
def load_model():
    return joblib.load("rf_churn_pipeline.pkl")

model = load_model()

st.title("📊 Customer Churn Prediction App (Random Forest)")
st.write("Predict customer churn using the trained Random Forest pipeline.")

# Initialize history if not exists
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure",
                 "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                 "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                 "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                 "MonthlyCharges", "TotalCharges", "Prediction", "Churn Probability"]
    )

# --- User Inputs ---
st.header("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

# --- Prediction ---
if st.button("Predict Churn"):
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
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

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to **Churn** (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Customer is likely to **Stay** (Probability: {1-proba:.2f})")

    # Append to history
    st.session_state.history.loc[len(st.session_state.history)] = [
        gender, senior, partner, dependents, tenure, phone_service, multiple_lines,
        internet_service, online_security, online_backup, device_protection, tech_support,
        streaming_tv, streaming_movies, contract, paperless_billing, payment_method,
        monthly_charges, total_charges,
        "Churn" if prediction == 1 else "Stay",
        round(proba, 2)
    ]

# --- Show History ---
st.subheader("📜 Prediction History")
if not st.session_state.history.empty:
    st.dataframe(st.session_state.history, use_container_width=True)

    # Download as CSV
    csv = st.session_state.history.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download History as CSV", csv, "churn_history.csv", "text/csv")
else:
    st.info("No predictions made yet.")
