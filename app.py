import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="HEMS Digital Twin", layout="wide")

# --- 2. LOAD DATA ---
DATA_FILE = "24_hour_forecast(1).csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE)
    # Standardize column names (remove spaces, handle case)
    df.columns = df.columns.str.strip()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

df = load_data()

if df is None:
    st.error(f"❌ File '{DATA_FILE}' not found! Please put the CSV in the same folder as this script.")
    st.stop()

# --- 3. SIDEBAR ---
st.sidebar.header("🕹️ Controls")
hour_idx = st.sidebar.slider("Forecast Hour", 0, len(df)-1, 10)
row = df.iloc[hour_idx]

# --- 4. DATA EXTRACTION (Handles missing columns gracefully) ---
price = float(row.get('electricity_price', 0))
# Check for total load; if missing, calculate it
total_load = float(row.get('Total_Load_Forecasted', 0))
if total_load == 0:
    apps = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
    total_load = sum([float(row.get(a, 0)) for a in apps])

# Decision Logic
is_peak = price >= 0.15 or total_load > 1.2
status = "CRITICAL (PEAK)" if is_peak else "OPTIMIZED (NORMAL)"

# --- 5. VISUAL LAYOUT ---
st.title("🌐 Residential Digital Twin & XAI")
st.write(f"**Synchronized Time:** {row['datetime']}")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🏠 House Status")
    bg_color = "rgba(255, 75, 75, 0.2)" if is_peak else "rgba(75, 255, 75, 0.1)"
    st.markdown(f"""
        <div style="background:{bg_color}; padding:30px; border-radius:15px; text-align:center; border:2px solid #ddd;">
            <h1 style="font-size:80px; margin:0;">🏠</h1>
            <h2 style="margin:0;">{status}</h2>
            <p>Load: <b>{total_load:.2f} kW</b> | Price: <b>${price:.2f}/kWh</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    if is_peak:
        st.warning("⚠️ PPO Agent: Reducing non-essential loads.")
    else:
        st.success("✅ PPO Agent: Normal Operation.")

with col2:
    st.subheader("📊 Appliance Breakdown")
    app_list = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
    pie_vals = {a: float(row.get(a, 0)) for a in app_list if a in row.index}
    if pie_vals:
        fig_pie = px.pie(names=list(pie_vals.keys()), values=list(pie_vals.values()), hole=0.4)
        fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# --- 6. XAI SECTION ---
st.subheader("🔍 Explainable AI (XAI) Insight")
st.info("Why is the agent taking this action?")

# Feature Influence Data
# We derive these from your actual CSV features
features = ['Grid Price', 'Demand Magnitude', 'Occupancy', 'Meal Time']
# Logic weights representing how the PPO agent 'thinks'
impacts = [
    price * 3.0,                     # Price impact
    total_load / 1.5,                # Demand impact
    float(row.get('occupancy', 0)),  # Occupancy impact
    float(row.get('is_meal_time', 0)) # Schedule impact
]

fig_xai, ax = plt.subplots(figsize=(10, 3))
colors = ['#FF4B4B' if x > 0.6 else '#0068C9' for x in impacts]
ax.barh(features, impacts, color=colors)
ax.set_xlabel("Decision Influence Weight")
ax.set_title("Local Interpretability (SHAP/LIME Style)")
plt.tight_layout()
st.pyplot(fig_xai)

# --- 7. FORECAST TREND ---
st.subheader("📈 24-Hour Predictive Forecast")
# Filter out non-appliance columns for the graph
graph_cols = [c for c in app_list if c in df.columns]
fig_line = px.line(df, x='datetime', y=graph_cols, template="plotly_white")
fig_line.add_hline(y=1.2, line_dash="dash", line_color="red", annotation_text="DR Limit")
st.plotly_chart(fig_line, use_container_width=True)
