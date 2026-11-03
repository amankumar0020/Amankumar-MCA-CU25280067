"""Streamlit symptom classifier. Not a diagnostic or medical decision tool."""
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd
BASE=Path(__file__).resolve().parent
SYMPTOMS=["fever","cough","headache","fatigue","sore_throat","runny_nose","nausea","vomiting",
"diarrhea","stomach_pain","rash","joint_pain","breathlessness","chest_pain","sneezing","loss_of_smell"]
st.set_page_config(page_title="Symptom Pattern Classifier",layout="centered")
st.title("Disease Prediction System — Educational Demo")
st.error("Not a medical diagnostic tool. The dataset is synthetic. Do not use predictions for treatment or health decisions.")
path=BASE/"models/disease_prediction_model.joblib"
if not path.exists():
    st.warning("Model not found. Run: python train.py"); st.stop()
model=joblib.load(path)
with st.form("symptom_form"):
    age=st.number_input("Age",min_value=1,max_value=110,value=25)
    temp=st.number_input("Body temperature (°C)",min_value=30.0,max_value=45.0,value=36.8,step=.1)
    st.write("Select symptoms that apply:")
    selected={s:st.checkbox(s.replace("_"," ").title()) for s in SYMPTOMS}
    submit=st.form_submit_button("Classify symptom pattern")
if submit:
    row={"age":age,"temperature_c":temp,**{s:int(selected[s]) for s in SYMPTOMS}}
    frame=pd.DataFrame([row])
    pred=model.predict(frame)[0]
    st.info(f"Model class: {pred}")
    if hasattr(model,"predict_proba"):
        probs=model.predict_proba(frame)[0]
        classes=model.classes_
        top=sorted(zip(classes,probs),key=lambda x:x[1],reverse=True)[:3]
        st.write("Top model classes (not clinical probabilities):")
        st.dataframe(pd.DataFrame(top,columns=["Model class","Model score"]))
    st.caption("If you feel unwell or have concerning symptoms, contact a qualified healthcare professional.")
