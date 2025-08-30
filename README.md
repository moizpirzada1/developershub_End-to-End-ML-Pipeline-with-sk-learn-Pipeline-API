📡 Telco Customer Churn Prediction

An end-to-end Machine Learning project to predict customer churn using the Telco Customer Churn dataset.
The project covers data preprocessing, model training, hyperparameter tuning, saving the best pipeline, and a beautiful Streamlit web app with prediction history.

🚀 Features

Data cleaning & preprocessing with Scikit-learn Pipelines

Models: Logistic Regression & Random Forest

Hyperparameter tuning using GridSearchCV

End-to-end Pipeline exported as .pkl

Streamlit UI for single & batch predictions

Prediction history stored in SQLite (viewable & downloadable)

Interactive probability gauge chart with Plotly

🗂 Project Structure
.
├── train_telco_churn.py       # Training script (builds and saves churn pipeline)
├── models/
│   └── churn_pipeline.pkl     # Saved ML pipeline
├── app.py                     # Streamlit app for predictions
├── history.db                 # SQLite DB storing prediction history
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

📊 Dataset

Telco Customer Churn dataset from IBM Sample Data
.

Each row represents a customer with attributes such as:

Demographics: Gender, Senior Citizen, Partner, Dependents

Services: Phone, Internet, Streaming, Tech Support

Account: Tenure, Contract type, Billing, Payment Method

Target: Churn (Yes=1, No=0)

⚙️ Installation

Clone this repo and install dependencies:

git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction
pip install -r requirements.txt

🏋️‍♂️ Train the Model

Run the training script (it will save the best model to models/churn_pipeline.pkl):

python train_telco_churn.py --csv "WA_Fn-UseC_-Telco-Customer-Churn.csv" --out models/churn_pipeline.pkl

🌐 Run the Streamlit App

Launch the app locally:

streamlit run app.py
Check out the live app in action here: 
https://lnkd.in/ds7YHYiX



🖼 App Features

Single Prediction

Enter customer details in a form

Predict churn & probability

Gauge chart for churn risk

Batch Prediction

Upload CSV with multiple customers

Download predictions with churn probability

History Tab

View all past predictions (saved in SQLite)

Filter & download history

📦 Requirements

See requirements.txt
:

pandas, numpy, scikit-learn, joblib

streamlit, plotly

📌 Future Improvements

Add SHAP explainability for feature importance per prediction

Deploy app on Streamlit Cloud / Docker / Heroku

Add email notifications for high churn-risk customers
