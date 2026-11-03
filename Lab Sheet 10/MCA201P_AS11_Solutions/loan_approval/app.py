"""Streamlit UI for the loan approval educational demo."""
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd
BASE=Path(__file__).resolve().parent
st.set_page_config(page_title="Loan Approval Predictor",layout="centered")
st.title("Loan Approval Prediction System")
st.warning("Academic demonstration only. This model uses synthetic data and must not be used for real lending decisions.")
model_path=BASE/"models/loan_approval_model.joblib"
if not model_path.exists():
    st.error("Model not found. Run: python train.py"); st.stop()
model=joblib.load(model_path)
with st.form("loan_form"):
    gender=st.selectbox("Gender",["Male","Female"])
    married=st.selectbox("Marital status",["Yes","No"])
    dependents=st.selectbox("Dependents",["0","1","2","3+"])
    education=st.selectbox("Education",["Graduate","Not Graduate"])
    employed=st.selectbox("Self employed",["Yes","No"])
    applicant=st.number_input("Applicant income (monthly)",min_value=0,value=5000)
    coapplicant=st.number_input("Co-applicant income (monthly)",min_value=0,value=0)
    amount=st.number_input("Loan amount (dataset units)",min_value=1,value=150)
    term=st.selectbox("Loan term (months)",[120,180,240,300,360])
    credit=st.selectbox("Credit history",["Good (1)","Not good (0)"])
    area=st.selectbox("Property area",["Rural","Semiurban","Urban"])
    submit=st.form_submit_button("Predict")
if submit:
    row=pd.DataFrame([{"Gender":gender,"Married":married,"Dependents":dependents,"Education":education,
       "Self_Employed":employed,"ApplicantIncome":applicant,"CoapplicantIncome":coapplicant,
       "LoanAmount":amount,"Loan_Amount_Term":term,"Credit_History":1 if credit.startswith("Good") else 0,
       "Property_Area":area}])
    result=int(model.predict(row)[0])
    if result==1: st.success("Model prediction: Likely Approved")
    else: st.error("Model prediction: Likely Not Approved")
    st.caption("This is a statistical classroom prediction, not a bank decision or financial advice.")
