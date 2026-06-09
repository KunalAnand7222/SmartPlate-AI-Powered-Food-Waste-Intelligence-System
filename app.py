"""
SmartPlate -- AI-Powered Food Waste Intelligence System
Streamlit Dashboard with 7 professional sections.

Run with: streamlit run app.py
"""

import sys
import os
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -- Ensure project root is on sys.path --
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import load_data, generate_restaurant_data
from src.analysis import (
    calculate_kpis,
    waste_heatmap_data,
    detect_anomalies,
    item_drilldown,
    waste_by_item,
    waste_by_category,
    waste_by_day_of_week,
    weekly_waste_cost,
    correlation_matrix,
    top_wasted_items,
    weekly_summary_text,
)
from src.ml_model import train_models, generate_forecast, plot_feature_importance
from src.ai_insights import get_ai_recommendations
from src.utils import format_inr, get_all_items, style_plotly_fig, COLORS


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SmartPlate - Food Waste Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# Theme State
# ─────────────────────────────────────────────

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

is_dark = st.session_state.theme == "dark"


# ─────────────────────────────────────────────
# Theme-aware color tokens
# ─────────────────────────────────────────────

if is_dark:
    T = {
        "bg_page": "#121212",
        "bg_sidebar": "#141414",
        "bg_card": "#1E1E1E",
        "bg_card_hover": "#262626",
        "border": "#333333",
        "border_hover": "#D4A574",
        "text_primary": "#FFFFFF",
        "text_secondary": "#D0D0D0",
        "text_muted": "#999999",
        "text_faint": "#777777",
        "accent_amber": "#E8B87D",
        "accent_coral": "#E8826A",
        "accent_sage": "#82C496",
        "accent_rust": "#D4635A",
        "accent_teal": "#6DB8C0",
        "accent_primary": "#D4A574",
        "exec_bg": "rgba(109, 184, 192, 0.08)",
        "exec_border": "#6DB8C0",
        "whatif_bg": "rgba(130, 196, 150, 0.08)",
        "whatif_border": "#82C496",
        "rec_border": "#D4A574",
        "rec_hover_border": "#E8826A",
        "rec_hover_bg": "#252525",
        "grid_color": "rgba(255,255,255,0.08)",
        "chart_text": "#D0D0D0",
        "chart_tick": "#B0B0B0",
        "chart_legend": "#C0C0C0",
        "heatmap_scale": ["#121212", "#D4A574", "#E8826A", "#D4635A"],
        "anomaly_marker_border": "#1E1E1E",
    }
else:
    T = {
        "bg_page": "#FAFAFA",
        "bg_sidebar": "#F0EDE8",
        "bg_card": "#FFFFFF",
        "bg_card_hover": "#F8F6F3",
        "border": "#E0DCD6",
        "border_hover": "#B8935F",
        "text_primary": "#1A1A1A",
        "text_secondary": "#444444",
        "text_muted": "#6B6B6B",
        "text_faint": "#999999",
        "accent_amber": "#B8935F",
        "accent_coral": "#C4614E",
        "accent_sage": "#4A8F5E",
        "accent_rust": "#B84A40",
        "accent_teal": "#3A8A92",
        "accent_primary": "#B8935F",
        "exec_bg": "rgba(58, 138, 146, 0.06)",
        "exec_border": "#3A8A92",
        "whatif_bg": "rgba(74, 143, 94, 0.06)",
        "whatif_border": "#4A8F5E",
        "rec_border": "#B8935F",
        "rec_hover_border": "#C4614E",
        "rec_hover_bg": "#FAF8F5",
        "grid_color": "rgba(0,0,0,0.06)",
        "chart_text": "#444444",
        "chart_tick": "#555555",
        "chart_legend": "#555555",
        "heatmap_scale": ["#FAFAFA", "#D4A574", "#E8826A", "#D4635A"],
        "anomaly_marker_border": "#FFFFFF",
    }

THEME_GRADIENT = ["#D4A574", "#82C496", "#E8826A", "#6DB8C0", "#E8B87D",
                   "#A68B60", "#A8C5A0", "#C07060", "#5EABB3", "#CBA87A"]


# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global font */
    html, body, [class*="css"] {{
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Main container */
    .block-container {{
        padding-top: 2rem;
        max-width: 1200px;
    }}

    /* ---- Page background ---- */
    .stApp {{
        background-color: {T['bg_page']};
    }}

    /* Header styling */
    .main-header {{
        font-family: 'DM Serif Display', Georgia, serif;
        color: {T['text_primary']};
        font-size: 2.6rem;
        font-weight: 400;
        letter-spacing: -0.3px;
        margin-bottom: 0;
        line-height: 1.15;
    }}
    .sub-header {{
        color: {T['text_muted']};
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0.4rem;
        margin-bottom: 2rem;
        letter-spacing: 0.3px;
    }}

    /* KPI Cards */
    .kpi-card {{
        background: {T['bg_card']};
        border: 1px solid {T['border']};
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
        text-align: left;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }}
    .kpi-card:hover {{
        border-color: {T['border_hover']};
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }}
    .kpi-label {{
        color: {T['text_muted']};
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.6rem;
    }}
    .kpi-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 500;
        margin: 0.1rem 0;
        line-height: 1.2;
    }}
    .kpi-value.amber {{ color: {T['accent_amber']}; }}
    .kpi-value.coral {{ color: {T['accent_coral']}; }}
    .kpi-value.sage {{ color: {T['accent_sage']}; }}
    .kpi-value.rust {{ color: {T['accent_rust']}; }}
    .kpi-value.teal {{ color: {T['accent_teal']}; }}
    .kpi-sub {{
        color: {T['text_faint']};
        font-size: 0.72rem;
        margin-top: 0.35rem;
        letter-spacing: 0.2px;
    }}

    /* Section Headers */
    .section-header {{
        font-family: 'DM Serif Display', Georgia, serif;
        color: {T['text_primary']};
        font-size: 1.4rem;
        font-weight: 400;
        margin: 2.5rem 0 0.6rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid {T['border']};
    }}
    .section-subtext {{
        color: {T['text_muted']};
        font-size: 0.82rem;
        margin-bottom: 1.2rem;
        margin-top: -0.3rem;
    }}

    /* AI Recommendation Cards */
    .rec-card {{
        background: {T['bg_card']};
        border: 1px solid {T['border']};
        border-left: 3px solid {T['rec_border']};
        border-radius: 0 10px 10px 0;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.7rem;
        transition: border-left-color 0.3s, background 0.3s;
    }}
    .rec-card:hover {{
        border-left-color: {T['rec_hover_border']};
        background: {T['rec_hover_bg']};
    }}
    .rec-title {{
        color: {T['text_primary']};
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }}
    .rec-detail {{
        color: {T['text_secondary']};
        font-size: 0.88rem;
        line-height: 1.65;
    }}

    /* Executive Summary */
    .exec-summary {{
        background: {T['exec_bg']};
        border-left: 3px solid {T['exec_border']};
        border-radius: 0 10px 10px 0;
        padding: 1.2rem 1.5rem;
        color: {T['text_secondary']};
        font-size: 0.9rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }}
    .exec-summary strong {{
        color: {T['text_primary']};
    }}

    /* What-If Results */
    .whatif-result {{
        background: {T['whatif_bg']};
        border-left: 3px solid {T['whatif_border']};
        border-radius: 0 10px 10px 0;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: {T['bg_sidebar']};
        border-right: 1px solid {T['border']};
    }}
    [data-testid="stSidebar"] .stMarkdown h2 {{
        font-family: 'DM Serif Display', Georgia, serif;
        color: {T['text_primary']};
        font-weight: 400;
    }}
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {{
        color: {T['text_secondary']};
    }}
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {T['text_primary']};
    }}
    [data-testid="stSidebar"] .stCaption {{
        color: {T['text_muted']};
    }}

    /* Metric delta styling */
    div[data-testid="stMetricDelta"] {{
        font-weight: 600;
    }}

    /* Hide Streamlit defaults */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.85rem;
    }}

    /* Expander styling */
    .streamlit-expanderHeader {{
        font-size: 0.9rem;
        color: {T['text_secondary']};
    }}

    /* Button styling */
    .stButton > button {{
        border: 1px solid {T['border']};
        border-radius: 8px;
        background: {T['bg_card']};
        color: {T['text_primary']};
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        border-color: {T['border_hover']};
        background: {T['bg_card_hover']};
    }}

    /* Selectbox / Input styling */
    .stSelectbox label, .stTextInput label, .stSlider label {{
        color: {T['text_secondary']} !important;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }}

    /* Divider override */
    hr {{
        border-color: {T['border']};
    }}

    /* Caption style */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {T['text_muted']} !important;
    }}

    /* Theme toggle button */
    .theme-toggle {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        background: {T['bg_card']};
        border: 1px solid {T['border']};
        color: {T['text_secondary']};
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.3px;
        cursor: pointer;
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Plotly styling helper
# ─────────────────────────────────────────────

def style_chart(fig):
    """Apply theme-aware styling to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["chart_text"], family="DM Sans, sans-serif", size=12),
        xaxis=dict(gridcolor=T["grid_color"], zeroline=False,
                   tickfont=dict(color=T["chart_tick"])),
        yaxis=dict(gridcolor=T["grid_color"], zeroline=False,
                   tickfont=dict(color=T["chart_tick"])),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=T["chart_legend"], size=11),
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# Data Loading (Cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_data():
    """Load or generate restaurant data."""
    return load_data()


@st.cache_data(show_spinner=False)
def get_trained_model(_orders_df, _waste_df):
    """Train models and cache results."""
    return train_models(_orders_df, _waste_df)


@st.cache_data(show_spinner=False)
def get_forecast(_orders_df, _waste_df):
    """Generate 7-day forecast and cache."""
    return generate_forecast(_orders_df, _waste_df)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## SmartPlate")
    st.caption("Food Waste Intelligence")
    st.markdown("---")

    # Theme toggle
    theme_label = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"
    if st.button(theme_label, key="theme_toggle", use_container_width=True):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

    st.markdown("---")

    restaurant_name = st.text_input(
        "Restaurant Name",
        value="The Grand Thali",
        help="Enter your restaurant name"
    )

    st.markdown("---")

    # Load data first to get date range
    orders_df, waste_df = load_all_data()
    min_date = orders_df["date"].min().date()
    max_date = orders_df["date"].max().date()

    st.markdown("**Date Range Filter**")
    date_range = st.date_input(
        "Select range",
        value=(max_date - timedelta(days=90), max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter all charts to this date range"
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask_orders = (orders_df["date"].dt.date >= start_date) & (orders_df["date"].dt.date <= end_date)
        mask_waste = (waste_df["date"].dt.date >= start_date) & (waste_df["date"].dt.date <= end_date)
        orders_filtered = orders_df[mask_orders].copy()
        waste_filtered = waste_df[mask_waste].copy()
    else:
        orders_filtered = orders_df.copy()
        waste_filtered = waste_df.copy()

    st.markdown("---")
    st.markdown("### Data Snapshot")
    st.markdown(f"**Records:** {len(orders_filtered):,}")
    st.markdown(f"**Items:** {orders_filtered['item_name'].nunique()}")
    st.markdown(f"**Categories:** {orders_filtered['category'].nunique()}")

    st.markdown("---")
    st.markdown(
        f"<p style='color: {T['text_faint']}; font-size: 0.75rem; text-align: center;'>"
        "SmartPlate AI v1.0<br>Waste Intelligence Platform</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────

st.markdown(f'<div class="main-header">{restaurant_name}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Actionable waste analytics and forecasting for smarter kitchen operations</div>',
    unsafe_allow_html=True
)


# ====================================================================
# SECTION 1: KPI Cards
# ====================================================================

st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">Latest week metrics compared to previous period</div>', unsafe_allow_html=True)

kpis = calculate_kpis(waste_filtered)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Waste This Week</div>
        <div class="kpi-value amber">{kpis['total_waste_kg']:,.1f} kg</div>
        <div class="kpi-sub">{kpis['week_range']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Waste Cost (INR)</div>
        <div class="kpi-value coral">{format_inr(kpis['total_waste_cost'])}</div>
        <div class="kpi-sub">Weekly expenditure on waste</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    change_color = "sage" if kpis['pct_change'] < 0 else "rust"
    change_arrow = "down" if kpis['pct_change'] < 0 else "up"
    change_label = "Improvement" if kpis['pct_change'] < 0 else "Needs attention"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Waste % Change vs Last Week</div>
        <div class="kpi-value {change_color}">{abs(kpis['pct_change']):.1f}% {change_arrow}</div>
        <div class="kpi-sub">{change_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Highest Waste Item</div>
        <div class="kpi-value teal" style="font-size: 1.2rem;">{kpis['worst_item']}</div>
        <div class="kpi-sub">Top offender this week</div>
    </div>
    """, unsafe_allow_html=True)


# ====================================================================
# SECTION 2: Waste Heatmap
# ====================================================================

st.markdown('<div class="section-header">Waste Heatmap</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">Average waste per item across days of the week</div>', unsafe_allow_html=True)

heatmap_data = waste_heatmap_data(waste_filtered)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns.tolist(),
    y=heatmap_data.index.tolist(),
    colorscale=[
        [0.0, T["heatmap_scale"][0]],
        [0.33, T["heatmap_scale"][1]],
        [0.66, T["heatmap_scale"][2]],
        [1.0, T["heatmap_scale"][3]],
    ],
    colorbar=dict(
        title=dict(text="Avg Waste (kg)", font=dict(color=T["chart_text"])),
        tickfont=dict(color=T["chart_tick"]),
    ),
    hovertemplate="Item: %{y}<br>Day: %{x}<br>Avg Waste: %{z:.2f} kg<extra></extra>",
))
fig_heatmap.update_layout(
    height=500,
    xaxis_title="Day of Week",
    yaxis_title="Menu Item",
)
style_chart(fig_heatmap)
st.plotly_chart(fig_heatmap, use_container_width=True)


# ====================================================================
# SECTION 3: 90-Day Waste Trend with Anomalies
# ====================================================================

st.markdown('<div class="section-header">Waste Trend with Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">Daily totals with rolling mean and statistical outliers</div>', unsafe_allow_html=True)

anomaly_df = detect_anomalies(waste_filtered)

fig_trend = go.Figure()

# Normal points
normal = anomaly_df[~anomaly_df["is_anomaly"]]
fig_trend.add_trace(go.Scatter(
    x=normal["date"], y=normal["waste_kg"],
    mode="lines",
    name="Daily Waste",
    line=dict(color=T["accent_amber"], width=1.5),
    fill="tozeroy",
    fillcolor="rgba(212, 165, 116, 0.08)" if is_dark else "rgba(184, 147, 95, 0.08)",
))

# Anomaly points
anomalies = anomaly_df[anomaly_df["is_anomaly"]]
fig_trend.add_trace(go.Scatter(
    x=anomalies["date"], y=anomalies["waste_kg"],
    mode="markers",
    name="Anomaly",
    marker=dict(color=T["accent_rust"], size=10, symbol="diamond",
                line=dict(color=T["anomaly_marker_border"], width=1.5)),
))

# Upper bound line
fig_trend.add_trace(go.Scatter(
    x=anomaly_df["date"], y=anomaly_df["upper_bound"],
    mode="lines",
    name="Anomaly Threshold",
    line=dict(color=T["accent_coral"], width=1, dash="dash"),
))

# Rolling mean
fig_trend.add_trace(go.Scatter(
    x=anomaly_df["date"], y=anomaly_df["rolling_mean"],
    mode="lines",
    name="14-Day Rolling Mean",
    line=dict(color=T["accent_sage"], width=2),
))

fig_trend.update_layout(
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="Total Daily Waste (kg)",
    xaxis_title="Date",
)
style_chart(fig_trend)
st.plotly_chart(fig_trend, use_container_width=True)

anomaly_count = anomalies.shape[0]
st.caption(f"**{anomaly_count} anomaly days** detected using rolling mean + 2 sigma threshold")


# ====================================================================
# SECTION 4: Item Drilldown
# ====================================================================

st.markdown('<div class="section-header">Item Drilldown</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">Prep vs orders vs leftover breakdown by month</div>', unsafe_allow_html=True)

all_items = sorted(orders_filtered["item_name"].unique().tolist())
selected_item = st.selectbox("Select a menu item", all_items, index=0)

drilldown = item_drilldown(orders_filtered, waste_filtered, selected_item)

fig_drill = go.Figure()

fig_drill.add_trace(go.Bar(
    x=drilldown["month"], y=drilldown["total_prep"],
    name="Prep Qty",
    marker_color=T["accent_amber"],
    marker_line_width=0,
))
fig_drill.add_trace(go.Bar(
    x=drilldown["month"], y=drilldown["total_orders"],
    name="Orders Qty",
    marker_color=T["accent_sage"],
    marker_line_width=0,
))
fig_drill.add_trace(go.Bar(
    x=drilldown["month"], y=drilldown["total_waste"],
    name="Leftover Qty",
    marker_color=T["accent_coral"],
    marker_line_width=0,
))

fig_drill.update_layout(
    barmode="group",
    height=420,
    yaxis_title="Quantity",
    xaxis_title="Month",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title=dict(text=f"{selected_item} - Monthly Breakdown",
               font=dict(size=15, color=T["text_primary"])),
)
style_chart(fig_drill)
st.plotly_chart(fig_drill, use_container_width=True)


# ====================================================================
# SECTION 5: 7-Day Forecast
# ====================================================================

st.markdown('<div class="section-header">7-Day Waste Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">ML-driven predictions for upcoming waste by item</div>', unsafe_allow_html=True)

with st.spinner("Training models and generating forecast..."):
    model_results = get_trained_model(orders_df, waste_df)
    forecast_df = get_forecast(orders_df, waste_df)

# Model metrics display
mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Best Model</div>
        <div class="kpi-value sage" style="font-size: 1.2rem;">{model_results['best_model_name']}</div>
    </div>
    """, unsafe_allow_html=True)

with mcol2:
    rf = model_results['rf_metrics']
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Random Forest</div>
        <div class="kpi-value amber" style="font-size: 1rem;">MAE: {rf['MAE']} | RMSE: {rf['RMSE']}</div>
    </div>
    """, unsafe_allow_html=True)

with mcol3:
    xgb = model_results['xgb_metrics']
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">XGBoost</div>
        <div class="kpi-value teal" style="font-size: 1rem;">MAE: {xgb['MAE']} | RMSE: {xgb['RMSE']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Forecast bar chart
fig_forecast = px.bar(
    forecast_df,
    x="date",
    y="forecast_waste_kg",
    color="item_name",
    barmode="group",
    color_discrete_sequence=THEME_GRADIENT,
    labels={"forecast_waste_kg": "Predicted Waste (kg)", "date": "Date", "item_name": "Item"},
)
fig_forecast.update_layout(
    height=500,
    legend=dict(
        orientation="h", yanchor="bottom", y=-0.5,
        font=dict(size=10),
    ),
    title=dict(text="Predicted Waste Per Item - Next 7 Days",
               font=dict(size=15, color=T["text_primary"])),
    xaxis_title="Forecast Date",
    yaxis_title="Waste (kg)",
)
style_chart(fig_forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# Feature importance
with st.expander("View Feature Importance"):
    fi_fig = plot_feature_importance(model_results["feature_importance_df"])
    # Apply theme-aware styling to the pre-built figure
    fi_fig.update_layout(
        title_font_color=T["text_primary"],
        font_color=T["chart_text"],
        xaxis=dict(gridcolor=T["grid_color"], tickfont=dict(color=T["chart_tick"])),
        yaxis=dict(gridcolor=T["grid_color"], tickfont=dict(color=T["chart_tick"])),
    )
    st.plotly_chart(fi_fig, use_container_width=True)


# ====================================================================
# SECTION 6: AI Recommendations
# ====================================================================

st.markdown('<div class="section-header">AI-Powered Recommendations</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">Data-driven suggestions to reduce waste and cut costs</div>', unsafe_allow_html=True)

# API Key input
api_key_input = st.text_input(
    "Grok API Key (Optional)", 
    type="password", 
    help="Enter your xAI Grok API Key to generate real-time AI insights. If exhausted or empty, it will use fallback."
)

# Session state for recommendations
if "ai_recs" not in st.session_state:
    summary_text = weekly_summary_text(orders_df, waste_df)
    st.session_state.ai_recs = get_ai_recommendations(summary_text, api_key_input)

if st.button("Refresh Recommendations", type="primary"):
    summary_text = weekly_summary_text(orders_df, waste_df)
    st.session_state.ai_recs = get_ai_recommendations(summary_text, api_key_input)
    st.rerun()

recs = st.session_state.ai_recs

# Executive Summary
st.markdown(f"""
<div class="exec-summary">
    <strong>Executive Summary</strong><br><br>
    {recs['executive_summary']}
</div>
""", unsafe_allow_html=True)

# Recommendation Cards
for i, rec in enumerate(recs["recommendations"]):
    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-title">{rec['title']}</div>
        <div class="rec-detail">{rec['detail']}</div>
    </div>
    """, unsafe_allow_html=True)


# ====================================================================
# SECTION 7: What-If Simulator
# ====================================================================

st.markdown('<div class="section-header">What-If Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">Model the impact of reducing prep quantities on waste and cost</div>', unsafe_allow_html=True)

sim_col1, sim_col2 = st.columns([1, 1])

with sim_col1:
    sim_item = st.selectbox("Select item for simulation", all_items, index=0, key="sim_item")
    reduction_pct = st.slider(
        "Prep Reduction %",
        min_value=5, max_value=50, value=20, step=5,
        help="Simulate reducing daily prep quantity by this percentage"
    )

with sim_col2:
    # Calculate impact
    item_waste = waste_filtered[waste_filtered["item_name"] == sim_item]
    item_orders = orders_filtered[orders_filtered["item_name"] == sim_item]

    if len(item_waste) > 0 and len(item_orders) > 0:
        avg_daily_prep = item_orders["prep_qty"].mean()
        avg_daily_orders = item_orders["orders_qty"].mean()
        avg_daily_waste_cost = item_waste["waste_cost_inr"].mean()

        # Reduced prep
        new_prep = avg_daily_prep * (1 - reduction_pct / 100)

        # If new prep >= orders, waste reduces proportionally
        # If new prep < orders, some orders are unfulfilled (lost revenue)
        if new_prep >= avg_daily_orders:
            waste_reduction_ratio = (avg_daily_prep - new_prep) / (avg_daily_prep - avg_daily_orders) if avg_daily_prep > avg_daily_orders else 0
            waste_reduction_ratio = min(waste_reduction_ratio, 1.0)
            new_daily_cost = avg_daily_waste_cost * (1 - waste_reduction_ratio)
            monthly_savings = (avg_daily_waste_cost - new_daily_cost) * 30
            lost_orders = 0
        else:
            new_daily_cost = 0
            monthly_savings = avg_daily_waste_cost * 30
            lost_orders = avg_daily_orders - new_prep

        fulfillment_msg = (
            f'<span style="color: {T["accent_rust"]};">Potential lost orders: <strong>{int(lost_orders)}</strong> units/day</span>'
            if lost_orders > 0
            else f'<span style="color: {T["accent_sage"]};">All orders can still be fulfilled</span>'
        )

        st.markdown(f"""
        <div class="whatif-result">
            <strong style="color: {T['text_primary']};">Projected Monthly Savings</strong><br><br>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 500; color: {T['accent_sage']};">
                {format_inr(monthly_savings)}
            </span>
            <span style="color: {T['text_faint']};"> / month</span><br><br>
            <span style="color: {T['text_secondary']}; font-size: 0.85rem;">
                Current avg prep: <strong>{avg_daily_prep:.0f}</strong> units/day<br>
                New prep: <strong>{new_prep:.0f}</strong> units/day<br>
                Avg orders: <strong>{avg_daily_orders:.0f}</strong> units/day<br>
                {fulfillment_msg}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No data available for the selected item in this date range.")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: {T['text_faint']}; font-size: 0.78rem; letter-spacing: 0.3px;'>"
    "SmartPlate v1.0 -- Food Waste Intelligence Platform<br>"
    "Reducing restaurant food waste, one prediction at a time."
    "</p>",
    unsafe_allow_html=True
)
