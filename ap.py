import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(page_title="Stock Price Forecast using ARIMA",
                   layout="wide")

st.title("📈 Stock Price Forecast using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if st.button("Run Forecast"):

    try:
        end_date = datetime.today()
        start_date = end_date.replace(year=end_date.year - 5)

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )

        if data.empty:
            st.error("No data found.")
            st.stop()

        df = data[['Close']].dropna()

        st.subheader("Last 5 Years Stock Price")

        st.line_chart(df['Close'])

        st.subheader("Historical Data")
        st.dataframe(df.tail())

        # ARIMA Model
        model = ARIMA(df['Close'], order=(5,1,0))
        model_fit = model.fit()

        target_date = pd.Timestamp("2027-06-30")

        future_days = (
            target_date - df.index[-1]
        ).days

        if future_days <= 0:
            st.warning(
                "June 2027 is already within the available data period."
            )
        else:

            forecast = model_fit.forecast(
                steps=future_days
            )

            forecast_index = pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=future_days,
                freq="D"
            )

            forecast_df = pd.DataFrame(
                forecast,
                index=forecast_index,
                columns=["Forecast"]
            )

            st.subheader("Forecasted Prices")

            combined = pd.concat(
                [
                    df.rename(columns={"Close":"Price"}),
                    forecast_df.rename(
                        columns={"Forecast":"Price"}
                    )
                ]
            )

            st.line_chart(combined)

            june_2027_price = forecast_df.loc[
                forecast_df.index >= "2027-06-01"
            ].iloc[-1]["Forecast"]

            st.success(
                f"Predicted Stock Price in June 2027: "
                f"${june_2027_price:.2f}"
            )

            st.subheader("Forecast Sample")
            st.dataframe(forecast_df.tail())

    except Exception as e:
        st.error(str(e))
