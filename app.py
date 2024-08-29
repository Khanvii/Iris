import streamlit as st  # Import Streamlit for creating the web app
import pandas as pd     # Import pandas for data manipulation
import numpy as np      # Import numpy for numerical operations
from prediction import predict  # Import the predict function from a local module named prediction

st.title("Classifying Iris Flowers")
st.markdown("Toy model to play to classify iris flowers into \
            setosa, versicolor, virginica")

st.header('Plant Features')
col1, col2 = st.columns(2)
with col1:
    st.text('Sepal characteristics')
sepal_l = st.slider("Sepal lenght (cm)", 1.0, 8.0, 0.5)
sepal_w = st.slider("Sepal width (cm)", 2.0, 4.4, 0.5)
with col2:
    st.text('Pepal characteristics')
petal_l = st.slider("Petal lenght (cm)", 1.0, 7.0, 0.5)
petal_w = st.slider("Petal width (cm)", 0.1, 2.5, 0.5)

st.button('Predict type of Iris')
          
# Prepare input data for prediction
input_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    
    # Make prediction using the predict function
result = predict(input_data)

    # Display the prediction result
st.text(f"Predicted Iris species: {result[0]}")

# In case no prediction has been made yet, display a placeholder message
if result is None:
    st.text("Awaiting input to predict the Iris species.")