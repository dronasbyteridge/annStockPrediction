import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
model = tf.keras.models.load_model('stock_price_model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App UI
st.title("Stock Price Prediction App")

# Stock ticker input
ticker_symbol = st.text_input("Enter a Stock Ticker Symbol (e.g., AAPL for Apple):", "AAPL")

# Date range input
start_date = st.date_input("Select Start Date", pd.to_datetime('2020-01-01'))
end_date = st.date_input("Select End Date", pd.to_datetime('2023-01-01'))

# Button to trigger data fetch and prediction
if st.button("Predict Next Day Closing Price"):
    try:
        # Fetch stock data
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        # Check if data is available
        if stock_data.empty:
            st.error("No data found for the given stock ticker and date range.")
        else:
            # Create moving averages and other features
            stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
            stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
            stock_data.dropna(inplace=True)

            # Prepare the input features for the model
            features = stock_data[['MA10', 'MA50', 'MA200', 'Volume']]
            scaled_features = scaler.transform(features)

            # Predict the next day's closing price
            last_row = np.array([scaled_features[-1]])
            predicted_price = model.predict(last_row)[0][0]

            # Display results
            st.success(f"The predicted next day closing price for {ticker_symbol} is ${predicted_price:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display the stock data
if st.checkbox("Show Historical Stock Data"):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    st.dataframe(stock_data)
