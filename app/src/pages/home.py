import base64
from pathlib import Path

import streamlit as st

@st.cache()
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def write():
    dataProcessor = st.session_state.dataProcessor

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
        st.title('Churn client prediction')

    with dataset:

        st.write(""" This is a small view to our working dataset. Our target is the 
        Attrition_Flag feature. Which says if a customer is have close it's bank 
        account ('Attrited Costumer') or not ('Existing Costumer').
        
        Those values are going to be change for '1' and '0' respectively
        """)
        st.subheader('Attritied clients Data Set ')
        st.write(dataProcessor.df_org.head())

        st.write(f""" This Data set is contains {dataProcessor.df_org.shape[0]} 
        samples and {dataProcessor.df_org.shape[1]} columns
        """)
