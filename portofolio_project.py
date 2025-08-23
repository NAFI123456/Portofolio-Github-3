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

high_feat_column, medium_feat_column, low_feat_column= st.columns(3,border=True)

# High Feature Importance
high_feat_column.subheader('Customer Information')

number_input_age = high_feat_column.number_input("Customer's Age: ", min_value=0, max_value=100,value=18 , step=1)

selectbox_geography = high_feat_column.selectbox('Customer Location: ', ['France', 'Spain', 'Germany'])

radio_gender = high_feat_column.radio("Customer's Gender : ", ['Male','Female'])


# Medium Feature Importance
medium_feat_column.subheader('Customer Activity')

number_input_tenure = medium_feat_column.number_input('Years Since Customer Joined: ', min_value=0, max_value=10)

number_input_numofproducts = medium_feat_column.number_input('Number Of Credit Card Products Owned: ', min_value=0, max_value=10)

radio_isactivemember = medium_feat_column.radio('Is Customer An Active Member: ', ['Yes', 'No'])

radio_hascrcard = medium_feat_column.radio('Does Customer Have Credit Card: ', ['Yes', 'No'])


# Low Feaature Importance
low_feat_column.subheader('Customer Financial Information')

number_input_balance = low_feat_column.number_input("Customer's Balance: ", min_value=0.0, max_value=500000.0, value=5000.0,step=5000.0)

slider_estimatedsalary = low_feat_column.slider('Customer Estimated Salary: ', min_value=0.0, max_value=250000.0, step=10000.0)

slider_credit_score = low_feat_column.slider("Customer's Credit Score: ", min_value=300,max_value=1000)

value_mapping = {'Yes':1, 'No':0}


data_interpret = pd.DataFrame({
                     "Customer's Age": [number_input_age],
                     "Customer's Location": [selectbox_geography],
                     "Customer's Gender": [radio_gender],
                     
                     
                     "Years Since Customer Registered": [number_input_tenure],
                     "Number of Credits Cards Used By Customer": [number_input_numofproducts],
                     "Customer Is An Active Member": [radio_isactivemember],
                     "Has Credit Card": [radio_hascrcard],
                     
                     "Customer's Balance": [number_input_balance],
                     "Customer Estimated Salary": [slider_estimatedsalary],
                     "Customer's Credit Score": [slider_credit_score],
                     })

st.markdown(
    "<h4 style='text-align: center;'>Upcoming Predicted Dataset for Interpretation</h4>", 
    unsafe_allow_html=True
)
st.table(data_interpret.reset_index(drop=True))

data = pd.DataFrame({'Age': [number_input_age],
                     'Geography': [selectbox_geography],
                     'Gender': [radio_gender],

                     'Tenure': [number_input_tenure],
                     'NumOfProducts': [number_input_numofproducts],
                     'IsActiveMember': [value_mapping[radio_isactivemember]],
                     'HasCrCard': [value_mapping[radio_hascrcard]],

                     'Balance': [number_input_balance],
                     'EstimatedSalary': [slider_estimatedsalary], 
                     'CreditScore': [slider_credit_score],
                     })

st.markdown(
    "<h4 style='text-align: center;'>Upcoming Predicted Dataset for Machine Learning</h4>", 
    unsafe_allow_html=True
)
st.table(data.reset_index(drop=True))

button = st.button("Predict", use_container_width=True)

if button:
    st.markdown("<h3 style='text-align: center;'>Prediction Based On Customer Features</h3>", unsafe_allow_html=True)
    pred, pred_proba = get_prediction(data)

    label_map = {0: "Customer Will Not Churn", 1: "Customer Will Churn"}
    
    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][pred[0]]
    proba_no_churn = pred_proba[0][0]
    proba_churn = pred_proba[0][1]

    st.markdown(f"<h4 style='text-align: center;'>Predicted: {label_pred}</h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Customer Loyalty Rate: {proba_no_churn:.0%}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Customer Churn Rate: {proba_churn:.0%}</p>", unsafe_allow_html=True)
