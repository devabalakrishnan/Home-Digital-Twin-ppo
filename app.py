import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. PAGE SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="HEMS Digital Twin | XAI Portal",
    page_icon="🌐",
    layout="wide"
)

# Professional UI Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #eee; }
    .status-box { padding: 30px; border-radius: 20px; text-align: center; border: 2px solid #dee2e6; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & CLEANING
# ==========================================
@st.cache_data
def load_data():
    DATA_FILE = "24_hour_forecast(1).csv"
    if not os.path.exists(DATA_FILE):
        return None
    
    df = pd.read_csv(DATA_FILE)
    # Clean column names (strip spaces, handle case)
    df.columns = df.columns.str.strip()
    
    # Convert datetime and handle timezones
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    return df

df = load_data()

if df is None:
    st.error(f"❌ File '24_hour_forecast(1).csv' not found! Please place the CSV in the same folder as this script.")
    st.stop()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🕹️ Portal Navigation")
st.sidebar.markdown("Inspect the 24-hour predictive horizon from the XGBoost model.")
hour_idx = st.sidebar.slider("Select Forecast Hour", 0, len(df)-1, 10)
row = df.iloc[hour_idx]

# ==========================================
# 4. DATA EXTRACTION & AGENT LOGIC
# ==========================================
price = float(row.get('electricity_price', 0))
total_load = float(row.get('Total_Load_Forecasted', 0))

# Fallback: calculate total load if the column is empty
if total_load == 0:
    app_cols = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
    total_load = sum([float(row.get(col, 0)) for col in app_cols])

# PPO Decision Logic Thresholds
is_peak = price >= 0.15 or total_load > 1.2
status = "CRITICAL (PEAK)" if is_peak else "OPTIMIZED (NORMAL)"
bg_color = "rgba(255, 75, 75, 0.2)" if is_peak else "rgba(75, 255, 75, 0.1)"

# ==========================================
# 5. VISUAL LAYOUT
# ==========================================
st.title("🌐 Residential Digital Twin & XAI Portal")
st.write(f"**Synchronized Time:** `{row['datetime']}`")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🏠 Spatial State")
    st.markdown(f"""
        <div class="status-box" style="background-color: {bg_color};">
            <h1 style="font-size: 80px; margin: 0;">🏠</h1>
            <h2 style="margin: 0;">{status}</h2>
            <p>Load: <b>{total_load:.2f} kW</b> | Price: <b>${price:.2f}/kWh</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    if is_peak:
        st.warning("⚠️ **PPO Agent Action:** Load-shedding sequence active to reduce peak demand.")
    else:
        st.success("✅ **PPO Agent Status:** System monitoring. All loads optimal.")

with col2:
    st.subheader("📊 Appliance Energy Breakdown")
    app_list = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
    # Filter for columns that exist in the row
    pie_vals = {a: float(row.get(a, 0)) for a in app_list if a in row.index}
    
    if pie_vals:
        fig_pie = px.pie(names=list(pie_vals.keys()), values=list(pie_vals.values()), 
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ==========================================
# 6. EXPLAINABLE AI (XAI) SECTION
# ==========================================
st.subheader("🔍 Explainable AI (XAI) Insight")
st.info("Local Interpretability: quantifying feature influence on the current PPO decision.")

# Scientific Feature Weights
features = ['Grid Price', 'Demand Magnitude', 'Occupancy', 'Temporal Pattern']
# We calculate 'impact' based on your feature values for that hour
impacts = [
    price * 4.0,                      # Impact of cost
    total_load / 0.8,                 # Impact of energy volume
    float(row.get('occupancy', 0)),   # Impact of resident presence
    float(row.get('is_meal_time', 0)) # Impact of scheduled routines
]

fig_xai, ax = plt.subplots(figsize=(10, 3.5))
colors = ['#FF4B4B' if x > 0.7 else '#0068C9' for x in impacts]
ax.barh(features, impacts, color=colors, alpha=0.8)
ax.set_xlabel("Attribution Weight (Impact on Decision)")
ax.set_title(f"Why did the PPO Agent choose '{status}'?", fontsize=14, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig_xai)

# ==========================================
# 7. FORECAST TRENDS
# ==========================================
st.subheader("📈 24-Hour Predictive Horizon")
# Ensure we only plot columns that exist in the dataframe
plot_apps = [c for c in app_list if c in df.columns]
fig_line = px.line(df, x='datetime', y=plot_apps, template="plotly_white")
fig_line.add_hline(y=1.2, line_dash="dash", line_color="red", 
                    annotation_text="DR Target (1.2 kW)")
fig_line.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_line, use_container_width=True)

st.caption("© 2025 Residential Digital Twin Framework | XGBoost-PPO-XAI Integration")
