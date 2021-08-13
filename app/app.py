import joblib
import streamlit as st

import package.awesome_streamlit as ast
import src.pages.eda
import src.pages.home
import src.pages.predict
from utils.Predictor import Predictor
from utils.dataProcessor import DataProcessor


@st.cache(allow_output_mutation=True)
def buildDataProcessor():
    return DataProcessor()


@st.cache(allow_output_mutation=True)
def buildPredictor():
    return joblib.load('utils/model/model.pkl')
    #return Predictor()


ast.core.services.other.set_logging_format()

PAGES = {
    "Home": src.pages.home,
    "EDA (Exploratory Data Analysis)": src.pages.eda,
    "Predict": src.pages.predict
}

if 'dataProcessor' not in st.session_state:
    st.session_state.dataProcessor = buildDataProcessor()

if 'predictor' not in st.session_state:
    st.session_state.predictor = joblib.load('utils/model/model.pkl')
    #st.session_state.predictor = buildPredictor()


def main():
    st.sidebar.title("CHURN PREDICTION APP")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)
    st.sidebar.title("About")
    st.sidebar.info(
        "This a classification machine learning model app to predict which type "
        "of clients have the more propensity to close their bank account and "
        "define their characteristics"
    )
    st.sidebar.title("Author")
    st.sidebar.info(
        """Jesus Bueno"""
    )


if __name__ == "__main__":
    main()
