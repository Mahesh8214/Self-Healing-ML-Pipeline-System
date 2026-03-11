import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.pipelines.prediction_pipeline import PredictPipeline


st.title("Diamond Price Prediction")

carat = st.number_input("Carat", value=0.7)
depth = st.number_input("Depth", value=62.0)
table = st.number_input("Table", value=55.0)
x = st.number_input("X", value=5.8)
y = st.number_input("Y", value=5.8)
z = st.number_input("Z", value=3.6)

cut = st.selectbox("Cut", ["Fair","Good","Very Good","Premium","Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

if st.button("Predict Price"):

    data = pd.DataFrame({
        "carat":[carat],
        "depth":[depth],
        "table":[table],
        "x":[x],
        "y":[y],
        "z":[z],
        "cut":[cut],
        "color":[color],
        "clarity":[clarity]
    })

    pipeline = PredictPipeline()
    prediction = pipeline.predict(data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")