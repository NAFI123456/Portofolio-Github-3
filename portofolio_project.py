import streamlit as st
import pandas as pd
import joblib


st.markdown(
    "<h1 style='text-align: center;'>CUSTOMER CHURN PREDICTOR</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>This App Predicts the Probability of Churn and Loyalty In a Customer</h4>", 
    unsafe_allow_html=True
)


model = joblib.load('model_joblib.joblib')

def get_prediction(data:pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

st.set_page_config(layout="wide")

# High Feature Importance
high_feat_column, medium_feat_column, low_feat_column= st.columns(3,border=True)
high_feat_column.subheader('High Feature Importances')
number_input_age = high_feat_column.number_input("Customer's Age: ", min_value=0, max_value=100,value=18 , step=1)

number_input_numofproducts = high_feat_column.number_input('Number Of Credit Card Products Owned: ', min_value=0, max_value=10)

number_input_balance = high_feat_column.number_input("Customer's Balance: ", min_value=0.00, max_value=500000.00, value=5000.00,step=5000.00)

# Medium Feature Importance
medium_feat_column.subheader('Medium Feature Importances')
slider_estimatedsalary = medium_feat_column.slider('Customer Estimated Salary: ', min_value=0.00, max_value=250000.00, step=10000.00)

radio_isactivemember = medium_feat_column.radio('Is Customer An Active Member: ', ['Yes', 'No'])

slider_credit_score = medium_feat_column.slider("Customer's Credit Score: ", min_value=300,max_value=1000)

selectbox_geography = medium_feat_column.selectbox('Customer Location: ', ['France', 'Spain', 'Germany'])


# Low Feaature Importance
low_feat_column.subheader('Low Feature Importances')
radio_gender = low_feat_column.radio("Customer's Gender : ", ['Male','Female'])

radio_hascrcard = low_feat_column.radio('Does Customer Have Credit Card: ', ['Yes', 'No'])

number_input_tenure = low_feat_column.number_input('Years Since Customer Joined: ', min_value=0, max_value=10)


value_mapping = {'Yes':1, 'No':0}


data = pd.DataFrame({'Age': [number_input_age],
                     'NumOfProducts': [number_input_numofproducts],
                     'Balance': [number_input_balance],

                     'EstimatedSalary': [slider_estimatedsalary],
                     'IsActiveMember': [value_mapping[radio_isactivemember]],
                     'CreditScore': [slider_credit_score],
                     'Geography': [selectbox_geography],

                     'Gender': [radio_gender],
                     'HasCrCard': [value_mapping[radio_hascrcard]],
                     'Tenure': [number_input_tenure],
                     })

st.markdown(
    "<h4 style='text-align: center;'>Upcoming Predicted Dataset</h4>", 
    unsafe_allow_html=True
)
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
