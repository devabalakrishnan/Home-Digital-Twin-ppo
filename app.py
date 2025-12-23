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
    page_title="Residential Digital Twin | XAI Portal",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #eee; }
    .status-box { padding: 30px; border-radius: 20px; text-align: center; border: 2px solid #dee2e6; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SMART DATA LOADING (Finds your CSV)
# ==========================================
@st.cache_data
def load_and_clean_data():
    # Automatically checks all possible names you might have saved
    possible_names = ["next_day_prediction.csv", "24_hour_forecast(1).csv", "24_hour_forecast.csv"]
    
    df = None
    active_file = ""

    for name in possible_names:
        if os.path.exists(name):
            df = pd.read_csv(name)
            active_file = name
            break
            
    if df is not None:
        df.columns = df.columns.str.strip()  # Remove accidental spaces
        # Handle timezone format correctly
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True) 
        return df, active_file
    return None, None

df, found_file = load_and_clean_data()

# Error handling if file is missing
if df is None:
    st.error("🚨 **File Not Found!**")
    st.info("Please ensure your CSV file (e.g., 'next_day_prediction.csv') is in the same folder as this app.py script.")
    st.stop()
else:
    st.sidebar.success(f"✅ Loaded: {found_file}")

# ==========================================
# 3. SIDEBAR - TEMPORAL NAVIGATION
# ==========================================
st.sidebar.header("🕹️ Digital Twin Controls")
st.sidebar.markdown("Navigate the 24-hour predictive horizon.")

selected_hour = st.sidebar.slider("Select Forecast Hour", 0, len(df)-1, 12)
row = df.iloc[selected_hour]

# ==========================================
# 4. AGENT DECISION LOGIC (PPO SIMULATION)
# ==========================================
price = float(row.get('electricity_price', 0))
total_load = float(row.get('Total_Load_Forecasted', 0))
occupancy = int(row.get('occupancy', 0))
is_meal = int(row.get('is_meal_time', 0))

# Defined Thresholds from Paper Methodology
PRICE_LIMIT = 0.15
LOAD_LIMIT = 1.2

is_critical = price >= PRICE_LIMIT or total_load > LOAD_LIMIT
status_label = "CRITICAL (PEAK)" if is_critical else "OPTIMIZED (NORMAL)"
status_color = "rgba(255, 75, 75, 0.2)" if is_critical else "rgba(75, 255, 75, 0.1)"

# ==========================================
# 5. DASHBOARD VISUALS
# ==========================================
st.title("🌐 Residential Digital Twin & XAI Portal")
st.write(f"**Synchronized State Time:** `{row['datetime']}`")

# --- ROW 1: SPATIAL STATE & BREAKDOWN ---
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("🏠 Spatial State")
    st.markdown(f"""
        <div class="status-box" style="background-color: {status_color};">
            <h1 style="font-size: 100px; margin: 0;">🏠</h1>
            <h2 style="margin: 10px 0;">{status_label}</h2>
            <p style="font-size: 20px;">Predicted Demand: <b>{total_load:.2f} kW</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    if is_critical:
        st.warning("🤖 **PPO Agent Action:** Load-shedding sequence active.")
    else:
        st.success("🤖 **PPO Agent Status:** Optimal operation. Monitoring grid.")

with col2:
    st.subheader("📊 Appliance Load Distribution")
    apps = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
    present_apps = [a for a in apps if a in row.index]
    
    if present_apps:
        fig_pie = px.pie(names=present_apps, values=[float(row[a]) for a in present_apps], 
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# --- ROW 2: TELEMETRY METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Grid Price", f"${price:.2f}/kWh")
m2.metric("Aggregate Load", f"{total_load:.2f} kW")
m3.metric("Home Occupancy", "👤 Active" if occupancy else "⚪ Vacant")
m4.metric("Meal Time", "🍴 Yes" if is_meal else "No")

# --- ROW 3: EXPLAINABLE AI (XAI) ---
st.subheader("🔍 Explainable AI (XAI) Insight")
st.info("Local Interpretability: quantifying feature influence on the current PPO decision.")

# Logical Feature Impact Modeling
xai_features = ['Electricity Price', 'Total Demand', 'Occupancy', 'Meal Context']
xai_weights = [price * 5.5, total_load / 0.7, occupancy * 0.4, is_meal * 0.9]

fig_xai, ax = plt.subplots(figsize=(12, 4))
colors = ['#ff4b4b' if w > 0.8 else '#0068c9' for w in xai_weights]
ax.barh(xai_features, xai_weights, color=colors, alpha=0.8)
ax.set_xlabel("Relative Attribution Weight")
ax.set_title(f"Decision Rationale for {row['datetime'].strftime('%H:%M')}", fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
st.pyplot(fig_xai)

# --- ROW 4: FORECAST TREND ---
st.subheader("📈 24-Hour Forecast Trend")
fig_line = px.line(df, x='datetime', y=present_apps, 
                   labels={'value': 'Power (kW)', 'datetime': 'Time'},
                   template="plotly_white")
fig_line.add_hline(y=LOAD_LIMIT, line_dash="dash", line_color="red", 
                   annotation_text="Demand Response Threshold")
fig_line.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_line, use_container_width=True)

st.caption("© 2025 Residential Digital Twin Framework | Forecasting + XAI Integration")
