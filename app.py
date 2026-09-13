import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gradio as gr

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

    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    return result, f"{probability:.1%}"

teal_theme = gr.themes.Soft(primary_hue="teal", secondary_hue="teal", neutral_hue="slate")

custom_css = """
:root, .dark {
    --body-background-fill: white !important;
    --background-fill-primary: white !important;
    --background-fill-secondary: #f0fdfa !important;
    --block-background-fill: white !important;
    --block-border-color: #5eead4 !important;
    --border-color-primary: #5eead4 !important;
    --body-text-color: #1f2937 !important;
    --body-text-color-subdued: #6b7280 !important;
    --block-label-text-color: #0d9488 !important;
    --block-title-text-color: #0d9488 !important;
    --input-background-fill: white !important;
    --checkbox-label-background-fill: white !important;
    --checkbox-label-background-fill-selected: #0d9488 !important;
    --checkbox-label-text-color-selected: white !important;
    --checkbox-background-color: white !important;
    --checkbox-background-color-selected: #0d9488 !important;
    --button-primary-background-fill: #0d9488 !important;
    --button-primary-background-fill-hover: #0f766e !important;
    --button-primary-text-color: white !important;
    --button-secondary-background-fill: white !important;
    --button-secondary-border-color: #5eead4 !important;
}
body, .gradio-container {
    background: white !important;
}
h1, .prose h1 {
    color: #0d9488 !important;
}
"""

force_light_js = """
() => {
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
}
"""

demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender", value="Male"),
        gr.Radio(["Yes", "No"], label="Married", value="Yes"),
        gr.Dropdown(["0", "1", "2", "3+"], label="Dependents", value="0"),
        gr.Radio(["Graduate", "Not Graduate"], label="Education", value="Graduate"),
        gr.Radio(["Yes", "No"], label="Self Employed", value="No"),
        gr.Number(label="Applicant Income", value=5000),
        gr.Number(label="Coapplicant Income", value=0),
        gr.Number(label="Loan Amount (in thousands)", value=120),
        gr.Number(label="Loan Amount Term (days)", value=360),
        gr.Radio([1, 0], label="Credit History (1=Good, 0=Bad)", value=1),
        gr.Radio(["Urban", "Semiurban", "Rural"], label="Property Area", value="Urban"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Approval Probability")
    ],
    title="🏦 Loan Approval Prediction App",
    description="Enter applicant details to predict loan approval using a trained Logistic Regression model.",
    theme=teal_theme,
    css=custom_css,
    js=force_light_js,
)

if __name__ == "__main__":
    demo.launch()
