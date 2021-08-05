import streamlit as st

from utils.dataProcessor import DataProcessor

st.write("""
# Churn client prediction""")

dataProcessor = DataProcessor()
dataProcessor.preprocess()