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

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .status-box { padding: 25px; border-radius: 15px; text-align: center; border: 1px solid #dee2e6; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. ROBUST DATA LOADING
# ==========================================
@st.cache_data
def load_and_clean_data():
    # List of possible filenames to avoid "File Not Found" errors
    possible_files = ["24_hour_forecast(1).csv", "24_hour_forecast.csv", "next_day_prediction.csv"]
    df = None
    
    for file in possible_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            break
            
    if df is not None:
        # Standardize Columns
        df.columns = df.columns.str.strip()
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            # Create dummy datetime if missing
            df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        return df
    return None

df = load_and_clean_data()

if df is None:
    st.error("🚨 **Error:** Data file not found. Please ensure '24_hour_forecast(1).csv' is in the same folder.")
    st.stop()

# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
st.sidebar.header("🕹️ Digital Twin Controls")
st.sidebar.markdown("Use the slider to inspect the 24-hour predictive horizon.")

# Slider to pick the hour
selected_hour = st.sidebar.slider("Select Forecast Hour", 0, len(df)-1, 12)
row = df.iloc[selected_hour]

# ==========================================
# 4. AGENT LOGIC (PPO SIMULATION)
# ==========================================
# Extract core variables
price = float(row.get('electricity_price', 0.10))
total_load = float(row.get('Total_Load_Forecasted', row.get('Total_Load', 0)))

# If Total_Load_Forecasted is 0, calculate it from appliances
if total_load == 0:
    apps = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
    total_load = sum([float(row.get(a, 0)) for a in apps])

# Optimization Logic (The Brain of your Paper)
price_threshold = 0.15 # Typical peak price
load_threshold = 1.5   # kW threshold
is_peak = price >= price_threshold or total_load > load_threshold

status_text = "CRITICAL (PEAK)" if is_peak else "OPTIMIZED (NORMAL)"
status_color = "rgba(255, 75, 75, 0.2)" if is_peak else "rgba(75, 255, 75, 0.1)"
agent_msg = "🚨 PPO ACTION: LOAD SHEDDING ACTIVE" if is_peak else "✅ PPO STATUS: MONITORING"

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================
st.title("🌐 Residential Digital Twin & XAI Portal")
st.write(f"**Temporal Sync:** {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")

# --- ROW 1: STATUS & TELEMETRY ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🏠 Spatial State")
    st.markdown(f"""
        <div class="status-box" style="background-color: {status_color};">
            <h1 style="font-size: 80px; margin: 0;">🏠</h1>
            <h2 style="margin: 5px;">{status_text}</h2>
            <p style="font-size: 18px;">Live Load: <b>{total_load:.2f} kW</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    if is_peak:
        st.warning(agent_msg)
    else:
        st.success(agent_msg)

with col2:
    st.subheader("📊 Appliance Consumption (kW)")
    app_list = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
    # Filter for columns that actually exist
    available_apps = [a for a in app_list if a in row.index]
    
    if available_apps:
        pie_df = pd.DataFrame({
            'Appliance': available_apps,
            'Load': [float(row[a]) for a
