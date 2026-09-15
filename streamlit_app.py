import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import streamlit as st

# ============================================================
# Page config (must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
)

# ============================================================
# Train the model (cached so it only runs once, not on every click)
# ============================================================
@st.cache_resource
def train_model():
    url = 'https://raw.githubusercontent.com/limchiahooi/loan-approval-prediction/master/train_u6lujuX_CVtuZ9i.csv'
    df = pd.read_csv(url)
    df = df.drop_duplicates()
    df = df.drop('Loan_ID', axis=1)

    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler, feature_columns


model, scaler, feature_columns = train_model()


def predict_loan(gender, married, dependents, education, self_employed,
                  applicant_income, coapplicant_income, loan_amount,
                  loan_term, credit_history, property_area):

    gender_val = 1 if gender == 'Male' else 0
    married_val = 1 if married == 'Yes' else 0
    education_val = 1 if education == 'Graduate' else 0
    self_employed_val = 1 if self_employed == 'Yes' else 0
    dependents_val = 3 if dependents == '3+' else int(dependents)
    property_urban = 1 if property_area == 'Urban' else 0
    property_semiurban = 1 if property_area == 'Semiurban' else 0

    input_dict = {
        'Gender': gender_val, 'Married': married_val, 'Dependents': dependents_val,
        'Education': education_val, 'Self_Employed': self_employed_val,
        'ApplicantIncome': applicant_income, 'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount, 'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area_Semiurban': property_semiurban, 'Property_Area_Urban': property_urban
    }
    input_df = pd.DataFrame([input_dict])[feature_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability


# ============================================================
# Custom CSS - white + teal theme
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: white;
    }
    h1 {
        color: #0d9488 !important;
    }
    .stButton>button {
        background-color: #0d9488;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 2em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #0f766e;
        color: white;
    }
    div[data-baseweb="radio"] label {
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# UI
# ============================================================
st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval using a trained Logistic Regression model.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Information")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    married = st.radio("Married", ["Yes", "No"], horizontal=True)
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.radio("Education", ["Graduate", "Not Graduate"], horizontal=True)
    self_employed = st.radio("Self Employed", ["Yes", "No"], horizontal=True)
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=120, step=10)
    loan_term = st.number_input("Loan Amount Term (days)", min_value=0, value=360, step=30)
    credit_history = st.radio("Credit History (1=Good, 0=Bad)", [1, 0], horizontal=True)
    property_area = st.radio("Property Area", ["Urban", "Semiurban", "Rural"], horizontal=True)

    predict_btn = st.button("Predict Loan Approval")

with col2:
    st.subheader("Prediction Result")

    if predict_btn:
        prediction, probability = predict_loan(
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_term, credit_history, property_area
        )

        if prediction == 1:
            st.success(f"### ✅ Approved")
        else:
            st.error(f"### ❌ Rejected")

        st.metric("Approval Probability", f"{probability:.1%}")
        st.progress(float(probability))
    else:
        st.info("👈 Fill in the applicant details and click **Predict Loan Approval** to see the result.")

st.divider()
st.caption("Model: Logistic Regression | Trained on the Loan Prediction dataset (614 records) | Accuracy: 86.2%")
