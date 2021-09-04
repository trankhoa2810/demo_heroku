import streamlit as st
from streamlit.elements.number_input import Number
from B1913240_TranAnhKhoa_train import training_Module


st.title("Iris flower sorting app.")

with st.form(key ='Form1'):
    sepalLength = st.number_input("sepal.length: ")
    sepalWidth = st.number_input("sepal.width: ")
    petalLength = st.number_input("petal.length: ")
    petalWidth = st.number_input("petal.width: ")
    submitted = st.form_submit_button(label = 'Prediction')
    if submitted:
        result = training_Module(sepalLength, sepalWidth, petalLength, petalWidth)
        st.write("Your Predicted Flower Class is: {}".format(result))