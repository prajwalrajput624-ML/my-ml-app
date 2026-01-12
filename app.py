import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Weather Prediction", page_icon="🌦")

# Load saved pipeline
pipe = pickle.load(open("weather.pkl", "rb"))
st.title("🌤 Weather Prediction App")

st.sidebar.header("Enter Weather Details")

input_data = {
    "Data.Precipitation": st.sidebar.number_input("Precipitation", min_value=0.0),
    "Date.Full": st.sidebar.text_input("Date (YYYY-MM-DD)"),
    "Date.Month": st.sidebar.number_input("Month", min_value=1, max_value=12),
    "Date.Week of": st.sidebar.number_input("Week of Year", min_value=1, max_value=53),
    "Date.Year": st.sidebar.number_input("Year", min_value=2000, max_value=2030),
    "Station.City": st.sidebar.text_input("Station City"),
    "Station.Code": st.sidebar.text_input("Station Code"),
    "Station.Location": st.sidebar.text_input("Station Location"),
    "Station.State": st.sidebar.text_input("Station State"),
    "Data.Temperature.Max Temp": st.sidebar.number_input("Max Temperature"),
    "Data.Temperature.Min Temp": st.sidebar.number_input("Min Temperature"),
    "Data.Wind.Direction": st.sidebar.text_input("Wind Direction"),
    "Data.Wind.Speed": st.sidebar.number_input("Wind Speed"),
}
if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = pipe.predict(input_df)
    st.success(f"🌡 Predicted Value: {prediction[0]:.2f}")
