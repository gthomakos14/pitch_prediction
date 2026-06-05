import streamlit as st
import requests
from config import API_URL
from components.sidebar import render_sidebar
from components.pitch_form import render_pitch_form
from components.prob_chart import render_prob_chart

st.set_page_config(page_title="Predict Next Pitch", page_icon="🎯", layout="wide")
render_sidebar()

st.title("Predict Next Pitch")
st.caption("Enter the game situation and get a pitch type probability distribution.")

with st.form("predict_form"):
    context = render_pitch_form(key_prefix="pred_")
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Calling model..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=context, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            st.success("Prediction ready")
            render_prob_chart(result.get("probabilities", {}))
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach API at `{API_URL}`. Start the server first.")
            st.code("uvicorn api.main:app --reload --port 8000", language="bash")
        except requests.exceptions.HTTPError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
