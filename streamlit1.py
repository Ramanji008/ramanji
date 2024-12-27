import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf  # For loading the .h5 model

# Load the trained model
model = tf.keras.models.load_model('model3.h5')

# Streamlit app
st.title("Real Estate Price Prediction")
st.write("Input property details to get a price prediction.")

# Input fields for user to provide property details
total_sqft = st.number_input("Total Square Feet", min_value=100, max_value=10000, value=1000, step=50)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1, step=1)
location = st.text_input("Location", placeholder="Enter the location")

# Predict button
if st.button("Predict Price"):
    # Prepare input data for the model
    # Here, location is a categorical feature; you may need to preprocess it (e.g., one-hot encoding) if the model expects it.
    input_data = np.array([[total_sqft, bath, balcony]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the predicted price
    st.subheader(f"Predicted Price for {location}: ₹{prediction[0][0]:,.2f}")

# Optional: Data insights
if st.checkbox("Show Data Insights"):
    st.write("Sample data used for training:")
    # Sample data display (replace with actual data if available)
    sample_data = pd.DataFrame({
        "Location": ["Location A", "Location B", "Location C"],
        "Total Sqft": [1000, 1500, 2000],
        "Bathrooms": [2, 3, 4],
        "Balconies": [1, 2, 3],
        "Price": [50, 75, 100],
    })
    st.table(sample_data)
