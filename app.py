
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from prophet import Prophet

# Load data
df = pd.read_csv("lotto_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Main number features
X = df[["d1","d2","d3","d4","d5","d6"]]
y = df[["d1","d2","d3","d4","d5","d6"]].shift(-1).dropna()

# Train model for next 6 numbers
model = RandomForestClassifier(n_estimators=100)
model.fit(X[:-1], y)

# Bonus ball model
bonus_y = df["bonus"].shift(-1).dropna().astype(int)
bonus_model = LogisticRegression(max_iter=200).fit(X[:-1], bonus_y)

# Prophet trends
stacked = df[["d1","d2","d3","d4","d5","d6"]].stack().reset_index(level=1, drop=True)
stacked = df[["d1", "d2", "d3", "d4", "d5", "d6"]].stack()
number_counts = stacked.value_counts().sort_index()

trend_forecasts = {}
for number in range(1, 39):
    ts = pd.DataFrame({
        "ds": df["date"],
        "y": (stacked == number).groupby(df["date"]).sum()
    })
    m = Prophet(weekly_seasonality=True)
    m.fit(ts)
    future = m.make_future_dataframe(periods=4)
    forecast = m.predict(future)
    trend_forecasts[number] = forecast["yhat"].iloc[-4:].mean()

top_trending = []
for num in range(1, 39):
    subset = ts_all[ts_all["number"] == num][["ds", "y"]]
    
    # Only run Prophet if at least 2 data points exist
    if subset["y"].sum() >= 2:
        m = Prophet(daily_seasonality=False, weekly_seasonality=True)
        m.fit(subset)
        future = m.make_future_dataframe(periods=4)
        forecast = m.predict(future)
        avg_forecast = forecast["yhat"].tail(4).mean()
        top_trending.append((num, avg_forecast))
