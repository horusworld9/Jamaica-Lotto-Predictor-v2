import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load real historical data
df = pd.read_csv("lotto_data.csv")
df.dropna(inplace=True)

# Prepare data
features = ["d1", "d2", "d3", "d4", "d5", "d6"]
X = df[features]
y_bonus = df["bonus"]

# Train Bonus Ball Model
bonus_model = LogisticRegression(max_iter=1000)
bonus_model.fit(X, y_bonus)

# Train model for predicting next set of numbers
main_model = RandomForestClassifier(n_estimators=200, random_state=42)
main_model.fit(X, X)

# Get latest draw as input for prediction
latest_draw = X.iloc[-1:]

# Generate multiple prediction sets
num_sets = 5
prediction_sets = []
for _ in range(num_sets):
    prediction = main_model.predict(latest_draw)[0]
    # Shuffle to simulate variability
    prediction = sorted(np.random.choice(prediction, size=6, replace=False))
    prediction_sets.append(prediction)

# Predict bonus ball
predicted_bonus = int(bonus_model.predict(latest_draw)[0])

# Streamlit UI
st.set_page_config(page_title="🇯🇲 Jamaica Lotto Predictor", layout="centered")
st.title("🇯🇲 Jamaica Lotto Predictor")
st.markdown("This app predicts the next **6 lotto numbers** and **bonus ball** using real draw data and machine learning.")

st.subheader("🎯 Predicted Sets")
for i, prediction in enumerate(prediction_sets, start=1):
    st.success(f"Set {i}: {prediction}")

st.subheader("🎁 Predicted Bonus Ball")
st.info(f"Bonus: {predicted_bonus}")
