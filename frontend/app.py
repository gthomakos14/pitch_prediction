import streamlit as st
from components.sidebar import render_sidebar

st.set_page_config(
    page_title="Pitch Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

st.title("MLB Pitch Type Predictor")
st.subheader("A neural network trained on Statcast data")

st.markdown("""
This app predicts the probability distribution over pitch types given the current
game situation. It calls a FastAPI backend backed by a PyTorch model.

---

### Pages

| Page | Description |
|------|-------------|
| **Predict Next Pitch** | Enter a game state manually and get an instant prediction |
| **At-Bat Explorer** | Build an at-bat pitch by pitch — watch predictions update as the sequence grows |

---

### How it works

1. **Data** — Statcast pitch-by-pitch data (2023–2026 seasons) fetched via `pybaseball`
2. **Features** — Game context: count, handedness, runners, inning, score, prior pitch sequence
3. **Model** — Feedforward MLP for per-pitch prediction; LSTM/Transformer for sequence-aware prediction
4. **API** — FastAPI service at `POST /predict`; returns per-class probabilities

Use the sidebar to navigate between pages or check API status.
""")
