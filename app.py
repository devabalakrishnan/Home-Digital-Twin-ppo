import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os

# 1. PAGE SETUP
st.set_page_config(page_title="Residential Digital Twin | XAI Portal", layout="wide")

# 2. DATA LOADING (Updated for your file structure)
@st.cache_data
def load_and_clean_data():
    # Looking in the 'data/' folder based on your screenshot
    possible_paths = ["data/next_day_prediction.csv", "next_day_prediction.csv"]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
            
    if df is not None:
        df.columns = df.columns.str.strip()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # MANDATORY: Calculate Total Load if missing
        app_cols = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
        df['Total_Load_Forecasted'] = df[app_cols].sum(axis=1)
        
        # Calculate dummy Meal Time for XAI (Price > 0.4 often coincides with meals)
        df['is_meal_time'] = (df['electricity_price'] > 0.4).astype(int)
        return df
    return None

df = load_and_clean_data()

if df is None:
    st.error("🚨 CSV File not found in 'data/' folder!")
    st.stop()

# 3. SIDEBAR
st.sidebar.header("🕹️ Controls")
selected_hour = st.sidebar.slider("Forecast Hour", 0, len(df)-1, 0)
row = df.iloc[selected_hour]

# 4. AGENT LOGIC
price = float(row['electricity_price'])
total_load = float(row['Total_Load_Forecasted'])
occupancy = int(row['occupancy'])
is_meal = int(row['is_meal_time'])

is_critical = price >= 0.4 or total_load > 2.0
status_label = "CRITICAL (PEAK)" if is_critical else "OPTIMIZED (NORMAL)"
status_color = "rgba(255, 75, 75, 0.2)" if is_critical else "rgba(75, 255, 75, 0.1)"

# 5. DASHBOARD VISUALS
st.title("🌐 Residential Digital Twin & XAI Portal")
st.write(f"**Current Sync Time:** `{row['datetime']}`")

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("🏠 Spatial State")
    st.markdown(f"""<div style="background:{status_color}; padding:30px; border-radius:15px; text-align:center; border:2px solid #ddd;">
        <h1 style="font-size:80px; margin:0;">🏠</h1><h2>{status_label}</h2>
        <p>Predicted Demand: <b>{total_load:.2f} kW</b></p></div>""", unsafe_allow_html=True)
    if is_critical: st.warning("🤖 PPO Agent: Initiating Load Shedding.")
    else: st.success("🤖 PPO Agent: Normal Operation.")

with col2:
    st.subheader("📊 Appliance Breakdown")
    apps = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
    fig_pie = px.pie(names=apps, values=[float(row[a]) for a in apps], hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ROW 2: METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("Grid Price", f"${price:.2f}")
m2.metric("Total Load", f"{total_load:.2f} kW")
m3.metric("Occupancy", f"{occupancy} Persons")
m4.metric("Meal Time", "Yes" if is_meal else "No")

# ROW 3: XAI
st.subheader("🔍 Explainable AI (XAI) Insight")
xai_features = ['Price', 'Total Demand', 'Occupancy', 'Meal Context']
xai_weights = [price * 2, total_load / 2, occupancy * 0.5, is_meal * 1.5]
fig_xai, ax = plt.subplots(figsize=(10, 3))
ax.barh(xai_features, xai_weights, color=['#ff4b4b' if w > 1 else '#0068c9' for w in xai_weights])
st.pyplot(fig_xai)

# ROW 4: TREND
st.subheader("📈 24-Hour Forecast")
fig_line = px.line(df, x='datetime', y=apps)
st.plotly_chart(fig_line, use_container_width=True)
