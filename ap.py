import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(page_title="Stock Forecasting App", layout="wide")

st.title("📈 Stock Price Forecasting using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if st.button("Generate Forecast"):

    try:
        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(years=5)

        st.write(f"Downloading 5 years of data for {ticker}...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True
        )

        if df.empty:
            st.error("No data found for this ticker.")
            st.stop()

        close_prices = df["Close"]

        st.subheader("Historical Stock Price")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(close_prices.index, close_prices)
        ax.set_title(f"{ticker} Historical Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)

        st.pyplot(fig)

        st.subheader("Training ARIMA Model")

        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()

        target_date = pd.Timestamp("2027-06-30")

        forecast_days = (target_date - close_prices.index[-1]).days

        if forecast_days <= 0:
            forecast_days = 365

        forecast = model_fit.forecast(steps=forecast_days)

        future_dates = pd.date_range(
            start=close_prices.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast.values
        })

        forecast_df.set_index("Date", inplace=True)

        june_data = forecast_df.loc["2027-06"]

        predicted_price = june_data["Forecast"].iloc[-1]

        st.success(
            f"Predicted Closing Price for June 2027: ${predicted_price:.2f}"
        )

        st.subheader("Forecast Chart")

        fig2, ax2 = plt.subplots(figsize=(14, 7))

        ax2.plot(
            close_prices.index,
            close_prices,
            label="Historical Data"
        )

        ax2.plot(
            forecast_df.index,
            forecast_df["Forecast"],
            label="ARIMA Forecast"
        )

        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig2)

        st.subheader("Forecast Data")
        st.dataframe(forecast_df.tail(30))

    except Exception as e:
        st.error(str(e))
