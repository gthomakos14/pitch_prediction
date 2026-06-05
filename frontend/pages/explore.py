import streamlit as st
import requests
from config import API_URL
from components.sidebar import render_sidebar
from components.pitch_form import render_pitch_form, PITCH_TYPES, PITCH_TYPE_CODES
from components.prob_chart import render_prob_chart

st.set_page_config(page_title="At-Bat Explorer", page_icon="🔍", layout="wide")
render_sidebar()

st.title("At-Bat Explorer")
st.caption(
    "Build an at-bat pitch by pitch. The model updates its next-pitch prediction "
    "as the sequence grows."
)

# ── Session state ─────────────────────────────────────────────────────────────
if "ab_context" not in st.session_state:
    st.session_state.ab_context = None
if "ab_history" not in st.session_state:
    st.session_state.ab_history: list[dict] = []
if "ab_prediction" not in st.session_state:
    st.session_state.ab_prediction: dict | None = None


def _fetch_prediction(context: dict, history: list[dict]) -> dict | None:
    payload = {
        **context,
        "pitch_number": len(history) + 1,
        "prev_pitches": [p["type"] for p in history],
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("probabilities")
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


# ── Game setup ────────────────────────────────────────────────────────────────
with st.expander("Game Setup", expanded=st.session_state.ab_context is None):
    with st.form("setup_form"):
        setup_ctx = render_pitch_form(key_prefix="setup_", prev_pitches=[])
        if st.form_submit_button("Start At-Bat", type="primary"):
            st.session_state.ab_context = setup_ctx
            st.session_state.ab_history = []
            st.session_state.ab_prediction = _fetch_prediction(setup_ctx, [])
            st.rerun()

# ── At-bat view ───────────────────────────────────────────────────────────────
if st.session_state.ab_context is None:
    st.info("Configure the game situation above to start an at-bat.")
    st.stop()

ctx = st.session_state.ab_context
history: list[dict] = st.session_state.ab_history

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("At-Bat State")

    matchup = f"{'RHP' if ctx['p_throws'] == 'R' else 'LHP'} vs {'RHB' if ctx['stand'] == 'R' else 'LHB'}"
    inning_str = f"{'Top' if ctx['inning_topbot'] == 'Top' else 'Bot'} {ctx['inning']}"
    score_str = f"Home {ctx['home_score']} – Away {ctx['away_score']}"
    runners = [base for base, key in [("1st", "on_1b"), ("2nd", "on_2b"), ("3rd", "on_3b")] if ctx.get(key)]

    st.markdown(f"**{matchup}** &nbsp;·&nbsp; {inning_str} &nbsp;·&nbsp; {score_str}")
    st.markdown(f"Runners: {', '.join(runners) if runners else 'None'} &nbsp;·&nbsp; Outs: {ctx['outs_when_up']}")

    if history:
        st.markdown("---")
        st.markdown("**Pitch history**")
        for i, pitch in enumerate(history, 1):
            label = PITCH_TYPES.get(pitch["type"], pitch["type"])
            st.markdown(f"{i}. **{pitch['type']}** {label} — {pitch['outcome']}")

    st.markdown("---")
    st.markdown(f"**Pitch #{len(history) + 1}** — throw a pitch:")

    pitch_type = st.selectbox(
        "Pitch type thrown",
        PITCH_TYPE_CODES,
        format_func=lambda c: f"{c} – {PITCH_TYPES.get(c, c)}",
        key="throw_type",
    )
    outcome = st.selectbox(
        "Outcome",
        ["Ball", "Called Strike", "Swinging Strike", "Foul", "In Play (out)", "In Play (hit)"],
        key="throw_outcome",
    )

    col_throw, col_reset = st.columns(2)
    with col_throw:
        if st.button("Throw Pitch", type="primary", use_container_width=True):
            st.session_state.ab_history.append({"type": pitch_type, "outcome": outcome})
            st.session_state.ab_prediction = _fetch_prediction(ctx, st.session_state.ab_history)
            st.rerun()
    with col_reset:
        if st.button("Reset", use_container_width=True):
            st.session_state.ab_context = None
            st.session_state.ab_history = []
            st.session_state.ab_prediction = None
            st.rerun()

with right:
    st.subheader(f"Model Prediction — Pitch #{len(history) + 1}")
    pred = st.session_state.ab_prediction
    if pred:
        render_prob_chart(pred)
    else:
        st.warning(f"API not reachable at `{API_URL}`. Predictions will appear once the server is running.")
        st.code("uvicorn api.main:app --reload --port 8000", language="bash")
