import streamlit as st
import pandas as pd
import altair as alt

PITCH_LABELS: dict[str, str] = {
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

# Fastballs → reds, breaking balls → blues, offspeed → greens
_PITCH_COLORS: dict[str, str] = {
    "FF": "#e63946",
    "SI": "#f4a261",
    "FC": "#e76f51",
    "SL": "#457b9d",
    "ST": "#1d3557",
    "CU": "#2a9d8f",
    "KC": "#264653",
    "CH": "#2d6a4f",
    "FS": "#52b788",
    "KN": "#95d5b2",
}


def render_prob_chart(probabilities: dict[str, float]) -> None:
    if not probabilities:
        st.info("No prediction to display.")
        return

    rows = [
        {
            "code": code,
            "label": PITCH_LABELS.get(code, code),
            "probability": prob,
            "color": _PITCH_COLORS.get(code, "#aaaaaa"),
        }
        for code, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    ]
    df = pd.DataFrame(rows)
    df["pct"] = (df["probability"] * 100).round(1)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("probability:Q", axis=alt.Axis(format="%", title="Probability"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color(
                "code:N",
                scale=alt.Scale(domain=list(_PITCH_COLORS.keys()), range=list(_PITCH_COLORS.values())),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Pitch Type"),
                alt.Tooltip("pct:Q", title="Probability (%)", format=".1f"),
            ],
        )
        .properties(height=max(200, len(rows) * 35))
    )
    st.altair_chart(chart, use_container_width=True)
