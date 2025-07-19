import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("lotto_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Prepare features and labels
X = df[["d1", "d2", "d3", "d4", "d5", "d6"]]
y_main = df[["d1", "d2", "d3", "d4", "d5", "d6"]].shift(-1).dropna()
y_bonus = df["bonus"].shift(-1).dropna().astype(int)

# Align feature rows with target rows
X_main = X.iloc[:-1]
X_bonus = X.iloc[:-1]

# Models
main_model = RandomForestClassifier(n_estimators=100, random_state=42)
main_model.fit(X_main, y_main)

bonus_model = LogisticRegression(max_iter=300)
bonus_model.fit(X_bonus, y_bonus)

# Get prediction from latest draw
latest_input = X.tail(1)
main_prediction = main_model.predict(latest_input)[0]
bonus_prediction = bonus_model.predict(latest_input)[0]

# UI
st.set_page_config(page_title="🇯🇲 Jamaica Lotto Predictor", layout="centered")
st.title("🇯🇲 Jamaica Lotto Predictor")
st.markdown("This app predicts the next 6 lotto numbers and bonus ball using machine learning.")

st.subheader("📊 Latest Prediction")
st.success(f"🎯 Predicted Numbers: {list(main_prediction)}")
st.info(f"🎁 Predicted Bonus Ball: {bonus_prediction}")
