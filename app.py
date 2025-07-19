import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load real Lotto history
df = pd.read_csv("lotto_data.csv", parse_dates=["date"])
X = df[["d1","d2","d3","d4","d5","d6"]]
y_main = df[["d1","d2","d3","d4","d5","d6"]].shift(-1).dropna()
y_bonus = df["bonus"].shift(-1).dropna().astype(int)
X_main, X_bonus = X.iloc[:-1], X.iloc[:-1]

# Train models
main_model = RandomForestClassifier(n_estimators=100, random_state=42)
main_model.fit(X_main, y_main)
bonus_model = LogisticRegression(max_iter=300)
bonus_model.fit(X_bonus, y_bonus)

# Predict probabilities for main numbers
latest = X.tail(1)
probs = main_model.predict_proba(latest)  # list of lists for each class
# For multi-output, we gather probability distribution per number
all_probs = np.mean(np.array([p[:,1] for p in probs]), axis=0)
# Generate 5 separate prediction sets
five_sets = []
for _ in range(5):
    pick = np.random.choice(range(1, 39), size=6, replace=False, p=all_probs/all_probs.sum())
    five_sets.append(sorted(map(int, pick)))

bonus_pred = int(bonus_model.predict(latest)[0])

# Display
st.set_page_config(page_title="🇯🇲 Jamaica Lotto Predictor", layout="centered")
st.title("🇯🇲 Jamaica Lotto Predictor")

st.subheader("🎲 Five Suggested Ticket Sets")
for i, s in enumerate(five_sets, 1):
    st.write(f"Set {i}: {', '.join(map(str, s))}")

st.subheader("🎁 Predicted Bonus Ball")
st.write(str(bonus_pred))
