import streamlit as st
import requests
from config import API_URL


def render_sidebar() -> None:
    with st.sidebar:
        st.title("⚾ Pitch Predictor")

        try:
            resp = requests.get(f"{API_URL}/health", timeout=2)
            if resp.status_code == 200:
                st.success("API online")
            else:
                st.warning(f"API status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("API offline")
            st.caption(f"Expected at `{API_URL}`")
            with st.expander("Start the API"):
                st.code("uvicorn api.main:app --reload --port 8000", language="bash")
        except Exception:
            st.error("API status unknown")

        st.divider()
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/predict.py", label="Predict Next Pitch", icon="🎯")
        st.page_link("pages/explore.py", label="At-Bat Explorer", icon="🔍")
        st.divider()
        st.caption(f"API: `{API_URL}`")
