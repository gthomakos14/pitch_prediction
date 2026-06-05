import streamlit as st

PITCH_TYPES: dict[str, str] = {
    "FF": "Four-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "KN": "Knuckleball",
}

PITCH_TYPE_CODES: list[str] = list(PITCH_TYPES.keys())


def _pitch_label(code: str) -> str:
    return f"{code} – {PITCH_TYPES.get(code, code)}"


def render_pitch_form(key_prefix: str = "", prev_pitches: list[str] | None = None) -> dict:
    """
    Render pitch context widgets. Returns a dict of field values ready to POST to /predict.
    If `prev_pitches` is provided, at-bat history is fixed (used by the explorer page).
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Matchup**")
        p_throws = st.selectbox("Pitcher Throws", ["R", "L"], key=f"{key_prefix}p_throws")
        stand = st.selectbox("Batter Stands", ["R", "L"], key=f"{key_prefix}stand")

    with col2:
        st.markdown("**Count & Outs**")
        balls = st.selectbox("Balls", [0, 1, 2, 3], key=f"{key_prefix}balls")
        strikes = st.selectbox("Strikes", [0, 1, 2], key=f"{key_prefix}strikes")
        outs = st.selectbox("Outs", [0, 1, 2], key=f"{key_prefix}outs")

    with col3:
        st.markdown("**Inning & Score**")
        inning = st.number_input("Inning", min_value=1, max_value=14, value=1, step=1, key=f"{key_prefix}inning")
        inning_topbot = st.selectbox("Half", ["Top", "Bot"], key=f"{key_prefix}inning_topbot")
        home_score = st.number_input("Home Score", min_value=0, max_value=30, value=0, step=1, key=f"{key_prefix}home_score")
        away_score = st.number_input("Away Score", min_value=0, max_value=30, value=0, step=1, key=f"{key_prefix}away_score")

    st.markdown("**Base Runners**")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        on_1b = st.checkbox("1st", key=f"{key_prefix}on_1b")
    with bc2:
        on_2b = st.checkbox("2nd", key=f"{key_prefix}on_2b")
    with bc3:
        on_3b = st.checkbox("3rd", key=f"{key_prefix}on_3b")

    if prev_pitches is None:
        st.markdown("**At-Bat History**")
        pitch_number = st.number_input(
            "Pitch # in at-bat", min_value=1, max_value=20, value=1, step=1,
            key=f"{key_prefix}pitch_number",
        )
        prev_pitches_val = st.multiselect(
            "Previous pitches (in order)",
            options=PITCH_TYPE_CODES,
            format_func=_pitch_label,
            key=f"{key_prefix}prev_pitches",
        )
    else:
        pitch_number = len(prev_pitches) + 1
        prev_pitches_val = prev_pitches

    return {
        "p_throws": p_throws,
        "stand": stand,
        "balls": balls,
        "strikes": strikes,
        "outs_when_up": outs,
        "inning": int(inning),
        "inning_topbot": inning_topbot,
        "home_score": int(home_score),
        "away_score": int(away_score),
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "pitch_number": int(pitch_number),
        "prev_pitches": prev_pitches_val,
    }
