import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import json
from datetime import datetime
import plotly.graph_objects as go

# ---------------------------
# Page Config & Styling
# ---------------------------
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="wide")

st.markdown("""
<style>
/* Center and card styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.card {
    padding: 1rem 1.5rem;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    background: white;
    border: 1px solid #eee;
}
.stButton>button {
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.pkl")

model = load_model()

# ---------------------------
# History Database
# ---------------------------
DB_PATH = "history.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            mode TEXT,
            prediction INTEGER,
            probability REAL,
            inputs TEXT
        )
        """)
init_db()

def log_history(mode, pred, prob, inputs):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO history (ts, mode, prediction, probability, inputs) VALUES (?, ?, ?, ?, ?)",
                     (datetime.now().isoformat(), mode, int(pred), float(prob) if prob is not None else None, json.dumps(inputs)))

def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    if not df.empty:
        df["inputs"] = df["inputs"].apply(lambda x: json.loads(x) if x else {})
        inputs_expanded = pd.json_normalize(df["inputs"])
        df = pd.concat([df.drop(columns=["inputs"]), inputs_expanded], axis=1)
    return df

# ---------------------------
# UI Layout
# ---------------------------
st.title("📡 Telco Churn Prediction App")
st.caption("End-to-end ML pipeline deployed with Streamlit | Tracks prediction history")

tabs = st.tabs(["🔮 Single Prediction", "📦 Batch Upload", "🗂 History"])

# ---------------------------
# Tab 1: Single Prediction
# ---------------------------
with tabs[0]:
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.checkbox("Senior Citizen?")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    if st.button("Predict Churn", type="primary"):
        input_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        df_input = pd.DataFrame([input_data])

        pred = model.predict(df_input)[0]
        try:
            prob = model.predict_proba(df_input)[:, 1][0]
        except:
            prob = None

        log_history("single", pred, prob, input_data)

        result = "Churn" if pred == 1 else "No Churn"
        st.success(f"Prediction: **{result}**")
        if prob is not None:
            st.write(f"Probability of churn: **{prob:.2f}**")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={"text": "Churn Probability (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Tab 2: Batch Upload
# ---------------------------
with tabs[1]:
    st.subheader("Upload CSV for Batch Prediction")
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        df_upload = pd.read_csv(file)
        preds = model.predict(df_upload)
        try:
            probs = model.predict_proba(df_upload)[:, 1]
        except:
            probs = np.zeros(len(preds))

        df_upload["Predicted_Churn"] = np.where(preds == 1, "Yes", "No")
        df_upload["Churn_Probability"] = probs

        st.dataframe(df_upload)

        # Log batch predictions
        for i, row in df_upload.iterrows():
            inputs = row.to_dict()
            log_history("batch", preds[i], probs[i], inputs)

        st.download_button("Download Predictions CSV",
                           df_upload.to_csv(index=False).encode("utf-8"),
                           "churn_predictions.csv",
                           "text/csv")

# ---------------------------
# Tab 3: History
# ---------------------------
with tabs[2]:
    st.subheader("Prediction History")

    df_hist = get_history()
    if df_hist.empty:
        st.info("No history yet. Make some predictions first.")
    else:
        st.dataframe(df_hist)
        st.download_button("Download History CSV",
                           df_hist.to_csv(index=False).encode("utf-8"),
                           "history.csv",
                           "text/csv")
