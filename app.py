
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
counts = stacked.groupby(df.index).value_counts().unstack(fill_value=0)

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

top_trending = sorted(trend_forecasts, key=trend_forecasts.get, reverse=True)[:6]

# UI
st.title("🇯🇲 Jamaica Lotto Predictor v2")
latest_input = X.tail(1)
prediction = model.predict(latest_input)[0]
bonus_prediction = bonus_model.predict(latest_input)[0]

st.success(f"Predicted Main Numbers: {list(prediction)}")
st.info(f"Predicted Bonus Ball: {bonus_prediction}")
st.success(f"Top Trending Numbers: {top_trending}")
