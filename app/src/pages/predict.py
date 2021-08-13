import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


@st.cache
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def write():
    dataProcessor = st.session_state.dataProcessor
    predictor = st.session_state.predictor

    header = st.beta_container()
    dataset = st.beta_container()
    features = st.beta_container()
    model_training = st.beta_container()

    with header:

        header_html = "<img src='data:image/png;base64,{}' class='img-fluid' width={}>".format(
            img_to_bytes("./src/assets/bank-churn.jpeg"), header.__sizeof__() * 22
        )
        st.markdown(
            header_html, unsafe_allow_html=True,
        )
        st.title('Make your predictions!')

    with dataset:
        st.write("""In order to make predictions about a possible churn, please
        fill in the following formulair and submit.
        """)
        prediction = dict()
        with st.form(key='my_form'):
            col1, col2, col3 = st.beta_columns(3)
            with col1:
                prediction['Customer_Age'] = st.text_input(label='Costumer age')
                marital_status = st.selectbox('Marital Status', ['Unknown', 'Married', 'Single', 'Divorced'])
                prediction['Marital_Status_' + marital_status] = 1
                prediction['Dependent_count'] = st.text_input(label='Number of dependents of the customer')
                prediction['Contacts_Count_12_mon'] = st.text_input(label='Contacts with bank (in a year)')

            with col2:
                gender = st.selectbox('Gender', ['Male', 'Female'])

                if gender == 'Male':
                    prediction['Gender'] = 0
                else:
                    prediction['Gender'] = 1
                prediction['Total_Relationship_Count'] = st.text_input(label='Total number of products held')
                prediction['Total_Trans_Amt'] = st.text_input(label= "Total amount of transactions made in the last "
                                                                     "year")
                prediction['Total_Amt_Chng_Q4_Q1'] = st.text_input(label='Change in transaction amount over the last '
                                                                         'year (Q4 over Q1)')

            with col3:
                education_level = st.selectbox('Education Level', ['Unknown', 'Uneducated', 'High School', 'Graduate',  'College', 'Post-Graduate', 'Doctorate'])
                prediction['Education_Level_' + education_level] = 1
                prediction['Total_Revolving_Bal'] = st.text_input(label='Total Revolving Value')
                prediction['Months_Inactive_12_mon'] = st.text_input(label='Month Inactive (in a year)')
                prediction['Total_Revolving_Bal'] = st.text_input(label='Total Revolving value')

            col4, col5 = st.beta_columns(2)
            with col4:
                income_category = st.selectbox('Income Category', ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'])
                prediction['Income_Category_' + income_category] = 1
                prediction['Total_Ct_Chng_Q4_Q1'] = st.text_input(label='Change in transaction number over the last '
                                                                        'year (Q4 over Q1).')
            with col5:
                prediction['Avg_Utilization_Ratio'] = st.text_input(label='(Account balance / Credit limit) in the '
                                                                          'last year year)')
                card_category = st.selectbox('Card Category',
                                                             dataProcessor.df['Card_Category'].unique())
                prediction['Card_Category_' + card_category] = 1
            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                np_array = np.asarray(list(prediction.values()))
                to_predict = pd.DataFrame(np_array.reshape(1, -1), columns=prediction.keys())
                st.write(to_predict)
                if predictor.predict(to_predict) == 0:
                    st.success("This customer is probably to stay")
                else:
                    st.warning('This customer could probably leave the bank')



