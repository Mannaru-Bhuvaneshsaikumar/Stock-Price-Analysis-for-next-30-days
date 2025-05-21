import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

# Set page configuration
st.set_page_config(page_title="Apple Stock Price Predictor", layout="wide")

# Title and description
st.title("📈 Apple Stock Price Predictor")
st.markdown("Predict future Apple stock closing prices using an LSTM model.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

df = load_data()

# Sidebar
st.sidebar.header("Configuration")
show_data = st.sidebar.checkbox("Show Raw Data", value=False)
predict_button = st.sidebar.button("Predict Next Day Price")

# Display raw data
if show_data:
    st.subheader("📊 Apple Stock Data (Latest Records)")
    st.dataframe(df.tail(50), use_container_width=True)

# Display closing price history
st.subheader("📉 Apple Stock - Last 100 Days Close Price")
st.line_chart(df["Close"].tail(100))

# Load LSTM model
@st.cache_resource
def load_trained_model():
    return load_model("lstm_model.h5")

model = load_trained_model()

# Scaling and prediction
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(df[["Close"]])

# Get the last 60 days of data
last_60_days = scaled_close[-60:]
X_test = np.reshape(last_60_days, (1, 60, 1))

# Predict
if predict_button:
    pred_price = model.predict(X_test)
    pred_price_unscaled = scaler.inverse_transform(pred_price)
    predicted_value = pred_price_unscaled[0][0]

    # Show prediction
    st.success(f"📅 Predicted Closing Price for Next Day: **${predicted_value:.2f}**")

    # Plot
    future_date = df.index[-1] + datetime.timedelta(days=1)
    fig, ax = plt.subplots()
    ax.plot(df.index[-100:], df["Close"].tail(100), label="Historical Close")
    ax.scatter(future_date, predicted_value, color='red', label="Predicted Next Day")
    ax.set_title("Apple Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built by Mannaru Bhuvaneshsaikumar | Powered by Streamlit + LSTM")
