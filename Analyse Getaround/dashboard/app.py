"""
GetAround — Streamlit Dashboard
Delay Analysis: helps the Product Manager choose the minimum delay threshold.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GetAround — Delay Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 GetAround — Rental Delay Analysis Dashboard")
st.markdown(
    "This dashboard helps the **Product Manager** decide the optimal **minimum delay** "
    "between two rentals, by exploring the trade-off between reducing friction and protecting revenue."
)

import os

# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "..", "get_around_delay_analysis.xlsx")
    if not os.path.exists(file_path):
        file_path = os.path.join(base_path, "get_around_delay_analysis.xlsx")
    df = pd.read_excel(file_path)
    return df

df = load_data()

# ─── Precompute consecutive pairs ─────────────────────────────────────────────
@st.cache_data
def prepare_consecutive(df):
    consec = df.dropna(subset=[
        "previous_ended_rental_id",
        "time_delta_with_previous_rental_in_minutes",
    ]).copy()
    prev = df[["rental_id", "delay_at_checkout_in_minutes"]].rename(
        columns={
            "rental_id": "previous_ended_rental_id",
            "delay_at_checkout_in_minutes": "prev_delay",
        }
    )
    consec = consec.merge(prev, on="previous_ended_rental_id", how="left")
    consec["time_delta"] = consec["time_delta_with_previous_rental_in_minutes"]
    consec["prev_delay_filled"] = consec["prev_delay"].fillna(0)
    consec["is_problematic"] = consec["prev_delay_filled"] > consec["time_delta"]
    return consec

consecutive = prepare_consecutive(df)

# ─── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Simulation Parameters")

threshold = st.sidebar.slider(
    "Minimum delay threshold (minutes)",
    min_value=0, max_value=720, value=60, step=30,
)

scope = st.sidebar.radio(
    "Scope",
    options=["all", "connect"],
    format_func=lambda x: "All cars" if x == "all" else "Connect cars only",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About the data:**")
st.sidebar.markdown(f"- Total rentals: **{len(df):,}**")
st.sidebar.markdown(f"- Consecutive pairs: **{len(consecutive):,}**")
st.sidebar.markdown(
    f"- Completed rentals: **{(df['state']=='ended').sum():,}**"
)

# ─── Filter by scope ──────────────────────────────────────────────────────────
if scope == "connect":
    consec_scope = consecutive[consecutive["checkin_type"] == "connect"]
    all_scope = df[df["checkin_type"] == "connect"]
else:
    consec_scope = consecutive
    all_scope = df

prob_scope = consec_scope[consec_scope["is_problematic"]]
total_prob = len(prob_scope)
total_rentals = len(all_scope)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
solved = int((prob_scope["prev_delay_filled"] <= threshold).sum())
affected = int((consec_scope["time_delta"] < threshold).sum())

pct_solved = solved / total_prob * 100 if total_prob > 0 else 0
pct_affected = affected / total_rentals * 100

st.markdown("---")
st.subheader(f"📊 KPIs — Threshold: **{threshold} min** | Scope: **{'All cars' if scope == 'all' else 'Connect only'}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rentals (scope)", f"{total_rentals:,}")
col2.metric("Problematic Cases", f"{total_prob:,}")
col3.metric(
    "Problems Solved",
    f"{solved:,}",
    f"{pct_solved:.1f}% of problematic cases",
    delta_color="normal",
)
col4.metric(
    "Rentals Affected",
    f"{affected:,}",
    f"⚠️ {pct_affected:.1f}% revenue at risk",
    delta_color="inverse",
)

# ─── Trade-off simulation chart ───────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Trade-off Simulation: Problems Solved vs Revenue at Risk")

thresholds_range = list(range(0, 721, 30))
sim_results = []
for t in thresholds_range:
    s = int((prob_scope["prev_delay_filled"] <= t).sum())
    a = int((consec_scope["time_delta"] < t).sum())
    sim_results.append({
        "threshold": t,
        "pct_solved": s / total_prob * 100 if total_prob > 0 else 0,
        "pct_affected": a / total_rentals * 100,
    })
df_sim = pd.DataFrame(sim_results)

fig_tradeoff = make_subplots(specs=[[{"secondary_y": True}]])
fig_tradeoff.add_trace(
    go.Scatter(
        x=df_sim["threshold"], y=df_sim["pct_solved"],
        name="% Problems Solved", line=dict(color="#00CC96", width=2), mode="lines+markers",
    ),
    secondary_y=False,
)
fig_tradeoff.add_trace(
    go.Scatter(
        x=df_sim["threshold"], y=df_sim["pct_affected"],
        name="% Revenue at Risk", line=dict(color="#EF553B", width=2), mode="lines+markers",
    ),
    secondary_y=True,
)
fig_tradeoff.add_vline(
    x=threshold, line_dash="dash", line_color="#636EFA",
    annotation_text=f"Selected: {threshold}min",
)
fig_tradeoff.update_layout(
    template="plotly_white",
    xaxis_title="Minimum Delay Threshold (minutes)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_tradeoff.update_yaxes(title_text="% Problems Solved", secondary_y=False, ticksuffix="%")
fig_tradeoff.update_yaxes(title_text="% Revenue at Risk", secondary_y=True, ticksuffix="%")
st.plotly_chart(fig_tradeoff, use_container_width=True)

# ─── Delay distribution ───────────────────────────────────────────────────────
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("⏱️ Distribution of Checkout Delays")
    ended = df[df["state"] == "ended"].dropna(subset=["delay_at_checkout_in_minutes"])
    fig_hist = px.histogram(
        ended, x="delay_at_checkout_in_minutes", color="checkin_type",
        nbins=70, range_x=[-300, 720],
        title="Checkout Delay Distribution by Checkin Type",
        labels={"delay_at_checkout_in_minutes": "Delay (min)", "checkin_type": "Checkin Type"},
        color_discrete_map={"mobile": "#636EFA", "connect": "#EF553B"},
        opacity=0.75, barmode="overlay",
        template="plotly_white",
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="On time")
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#636EFA",
                       annotation_text=f"Threshold: {threshold}min")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    st.subheader("📦 Delay by Checkin Type (Box Plot)")
    fig_box = px.box(
        ended, x="checkin_type", y="delay_at_checkout_in_minutes",
        color="checkin_type", points="outliers",
        title="Checkout Delay by Checkin Type",
        color_discrete_map={"mobile": "#636EFA", "connect": "#EF553B"},
        template="plotly_white",
    )
    fig_box.add_hline(y=0, line_dash="dash", line_color="black")
    fig_box.add_hline(y=threshold, line_dash="dot", line_color="#636EFA",
                      annotation_text=f"Threshold: {threshold}min")
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# ─── State & checkin type breakdown ───────────────────────────────────────────
st.markdown("---")
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("📊 Rental State by Checkin Type")
    state_counts = df.groupby(["checkin_type", "state"]).size().reset_index(name="count")
    fig_state = px.bar(
        state_counts, x="checkin_type", y="count", color="state", barmode="group",
        title="Rental State Distribution",
        color_discrete_map={"ended": "#00CC96", "canceled": "#FF6692"},
        template="plotly_white",
    )
    st.plotly_chart(fig_state, use_container_width=True)

with col_d:
    st.subheader("⏳ Time Delta Between Consecutive Rentals")
    fig_delta = px.histogram(
        consec_scope, x="time_delta", color="checkin_type",
        nbins=50, title="Time Delta (Consecutive Rentals)",
        labels={"time_delta": "Time Delta (min)", "checkin_type": "Checkin Type"},
        color_discrete_map={"mobile": "#636EFA", "connect": "#EF553B"},
        opacity=0.75, barmode="overlay", template="plotly_white",
    )
    fig_delta.add_vline(x=threshold, line_dash="dash", line_color="#636EFA",
                        annotation_text=f"Threshold: {threshold}min")
    st.plotly_chart(fig_delta, use_container_width=True)

# ─── Raw data explorer ────────────────────────────────────────────────────────
with st.expander("🔍 Explore Raw Data"):
    st.dataframe(df.head(100), use_container_width=True)

st.markdown("---")
st.caption("Dashboard built with Streamlit · GetAround Delay Analysis · Jedha Bloc 5")
