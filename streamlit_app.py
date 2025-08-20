import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")

model = load_model()

st.title("📊 Customer Churn Prediction App")
st.write("This app predicts whether a customer will **churn** based on their details.")

# Initialize history if not exists
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["Gender", "SeniorCitizen", "Partner", "Dependents",
                 "Tenure", "MonthlyCharges", "TotalCharges", "Prediction", "Churn Probability"]
    )

# --- Inputs (adjust feature names as per your dataset) ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

if st.button("Predict Churn"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to Churn (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Customer is likely to Stay (Probability: {1-proba:.2f})")

    # Append to history
    st.session_state.history.loc[len(st.session_state.history)] = [
        gender, senior, partner, dependents, tenure,
        monthly_charges, total_charges, "Churn" if prediction == 1 else "Stay", round(proba, 2)
    ]

# --- Show History ---
st.subheader("📜 Prediction History")
if not st.session_state.history.empty:
    st.dataframe(st.session_state.history, use_container_width=True)
else:
    st.info("No predictions made yet.")
