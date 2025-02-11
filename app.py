import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the model's feature columns
with open("model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

# Streamlit UI
st.title("🚗 Car Price Prediction App")
st.write("Enter car details to predict the selling price.")

# User Inputs
year = st.number_input("Car Manufacturing Year", min_value=2000, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner = st.selectbox("Previous Owners", ["First Owner", "Second Owner", "Third Owner", "More than Three"])
engine = st.number_input("Engine Capacity (cc)", min_value=600, max_value=5000, value=1500)
max_power = st.number_input("Max Power (bhp)", min_value=30, max_value=500, value=100)
seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)

# Convert categorical values to match training
user_input = pd.DataFrame([[year, km_driven, engine, max_power, seats, fuel_type, transmission, owner]],
                          columns=["year", "km_driven", "engine", "max_power", "seats", "fuel_type", "transmission", "owner"])

# Apply one-hot encoding (same as training)
user_input = pd.get_dummies(user_input, drop_first=True)

# Ensure input has the same columns as the trained model
user_input = user_input.reindex(columns=model_columns, fill_value=0)

# Show result when "Predict" button is clicked
if st.button("Predict"):
    prediction = model.predict(user_input)
    st.success(f"💰 Predicted Selling Price: ${prediction[0]:,.2f}")
