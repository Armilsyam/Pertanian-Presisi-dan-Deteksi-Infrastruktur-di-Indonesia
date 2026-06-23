# app_streamlit.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="ML Studio Wow - Demo", layout="wide")
st.title("ML Studio Wow Demo")

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    target = st.text_input("Target column name")
    if target and target in df.columns:
        if st.button("Train locally quick"):
            st.write("Training (quick demo)...")
            # call train.py programmatically or run a small model inline
            st.success("Done. For full training use train.py on VM.")
st.markdown("Use the FastAPI service for production inference.")
