import streamlit as st
import pandas as pd
import joblib

st.title("Churn predictor")
st.write("This app predicts the probability of churn in a customer")


model = joblib.load('model_joblib.joblib')

def get_prediction(data:pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba



column_credit_score, = st.columns(1, border=True)
column_credit_score.subheader('Credit Score')
slider_credit_score = column_credit_score.slider("Customer's Credit Score Value: ", min_value=1.0,max_value=1000.0)

column_geography, = st.columns(1, border=True)
column_geography.subheader('Geography')
selectbox_geography = column_geography.selectbox('Customer Location: ', ['France', 'Spain', 'Germany'])

column_gender, = st.columns(1, border=True)
column_gender.subheader('Gender')
radio_gender = column_gender.radio("Customer's Gender : ", ['Male','Female'])

column_age, column_tenure = st.columns(2, border=True)
column_age.subheader('Age')
number_input_age = column_age.number_input("Customer's Age: ", min_value=0, max_value=100, step=1)

column_tenure.subheader('Tenure')
number_input_tenure = column_tenure.number_input('Years Since Customer Joined: ', min_value=0, max_value=10)

column_balance, = st.columns(1, border=True)
column_balance.subheader('Balance')
number_input_balance = column_balance.number_input("Customer's Balance: ", min_value=1, max_value=500000, value=500,step=10000)

column_numofproducts, = st.columns(1, border=True)
column_numofproducts.subheader('Number Of Products')
number_input_numofproducts = column_numofproducts.number_input('Number Of Credit Card Products Owned: ', min_value=0, max_value=10)

column_hascrcard, = st.columns(1, border=True)
column_hascrcard.subheader('Credit Card Ownership')
radio_hascrcard = column_hascrcard.radio('Does Customer Have Credit Card: ', ['Yes', 'No'])

column_isactivemember, = st.columns(1, border=True)
column_isactivemember.subheader('Active Member')
radio_isactivemember = column_isactivemember.radio('Is Customer An Active Member: ', ['Yes', 'No'])

column_estimatedsalary, = st.columns(1, border=True)
column_estimatedsalary.subheader('Estimated Salary')
slider_estimatedsalary = column_estimatedsalary.slider('Customer Estimated Salary: ', min_value=0, max_value=250000, step=10000)

value_mapping = {'Yes':1, 'No':0}



data = pd.DataFrame({'CreditScore': [slider_credit_score],
                     'Geography': [selectbox_geography],
                     'Gender': [radio_gender],
                     'Age': [number_input_age],
                     'Tenure': [number_input_tenure],
                     'Balance': [number_input_balance],
                     'NumOfProducts': [number_input_numofproducts],
                     'HasCrCard': [value_mapping[radio_hascrcard]],
                     'IsActiveMember': [value_mapping[radio_isactivemember]],
                     'EstimatedSalary': [slider_estimatedsalary]
                     })

st.dataframe(data)

button = st.button("Predict", use_container_width=True)

if button:
    st.markdown('**Prediction Based On Customer Features**')
    pred, pred_proba = get_prediction(data)

    label_map = {0: "Customer Will Not Churn", 1: "Customer Will Churn"}
    
    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][pred[0]]
    proba_no_churn = pred_proba[0][0]
    proba_churn = pred_proba[0][1]

    st.write(f"Predicted: {label_pred}")
    st.write(f"Customer Loyality Rate: {proba_no_churn:.0%}")
    st.write(f"Customer Churn Rate: {proba_churn:.0%}")
