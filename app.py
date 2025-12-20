import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import plotly.express as px

# --- 1. SETTINGS & PATHS ---
st.set_page_config(page_title="Home Digital Twin AI", layout="wide")

# Folder paths
DATA_PATH = os.path.join('data', 'next_day_prediction.csv')
ASSET_PATH = os.path.join('assets', 'house_model.svg')

# --- 2. ASSET & VISUALIZATION FUNCTIONS ---
def get_base64_of_bin_file(bin_file):
    """Converts local file to base64 for HTML display."""
    if not os.path.exists(bin_file):
        return None
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

def display_custom_house(load_status, load_value):
    """Displays the house model with dynamic HTML/CSS."""
    # Ensure load_value is a simple number
    val = float(load_value)
    
    # Define colors
    glows = {
        "CRITICAL": "rgba(255, 0, 0, 0.4)", 
        "HIGH": "rgba(255, 165, 0, 0.4)", 
        "OPTIMIZED": "rgba(0, 255, 0, 0.2)"
    }
    glow_color = glows.get(str(load_status), glows["OPTIMIZED"])

    img_base64 = get_base64_of_bin_file(ASSET_PATH)
    
    if img_base64:
        # Use your uploaded SVG
        img_html = f'<img src="data:image/svg+xml;base64,{img_base64}" width="100%" style="max-width: 350px;">'
    else:
        # Use a high-quality online icon if local file is missing
        url = "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/house-chimney.svg"
        img_html = f'<img src="{url}" width="120" style="filter: invert(0.5); opacity: 0.8;">'

    # FIXED: Changed unsafe_allow_code to unsafe_allow_html
    st.markdown(
        f"""
        <div style="text-align: center; padding: 40px; background: {glow_color}; border-radius: 25px; border: 2px solid rgba(255,255,255,0.1);">
            {img_html}
            <div style="margin-top: 20px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 10px;">
                <h3 style="margin:0; color: black;">Live Load: {val:.2f} kW</h3>
                <p style="margin:0; color: #333;">State: <b>{str(load_status)}</b></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 3. MAIN APP LOGIC ---
st.title("🌐 Residential Digital Twin Portal")

# Check if data exists
if not os.path.exists(DATA_PATH):
    st.error(f"🚨 Data file not found at `{DATA_PATH}`. Please run your training script first.")
else:
    df = pd.read_csv(DATA_PATH)
    
    # Simulation Sidebar
    st.sidebar.header("🕹️ Controls")
    hour_idx = st.sidebar.slider("Select Forecast Hour", 0, len(df)-1, 0)
    row = df.iloc[hour_idx]

    # --- 4. DATA EXTRACTION & PPO LOGIC ---
    # Convert pandas values to standard Python floats
    price = float(row.get('electricity_price', 0))
    occupancy = float(row.get('occupancy', 0))
    
    # Exact appliance names from your CSV
    appliances = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing Machine']
    available_apps = [app for app in appliances if app in row.index]
    
    # Calculate loads
    total_raw_load = float(row[available_apps].sum())

    # Simulated PPO Agent Decision
    if price > 0.6:
        status = "CRITICAL"
        ppo_msg = "PPO ACTION: LOAD SHEDDING"
        # Reduce heavy loads (Heater/Washing Machine) by 70%
        reduction = (float(row.get('Heater', 0)) * 0.7) + (float(row.get('Washing Machine', 0)) * 0.7)
        optimized_load = total_raw_load - reduction
    elif price > 0.4:
        status = "HIGH"
        ppo_msg = "PPO STATUS: MONITORING"
        optimized_load = total_raw_load
    else:
        status = "OPTIMIZED"
        ppo_msg = "PPO STATUS: NORMAL"
        optimized_load = total_raw_load

    # --- 5. DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("🏠 Spatial Synchronization")
        display_custom_house(status, optimized_load)
        st.info(f"**Agent Decision:** {ppo_msg}")

    with col2:
        st.subheader("📊 Energy Telemetry")
        if available_apps:
            # Create a pie chart
            pie_values = [float(row[app]) for app in available_apps]
            fig = px.pie(values=pie_values, names=available_apps, hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No appliance data available for this hour.")

    st.divider()
    
    # Financial/Efficiency Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Load", f"{total_raw_load:.2f} kW")
    m2.metric("Optimized Load", f"{float(optimized_load):.2f} kW", f"{float(optimized_load - total_raw_load):.2f} kW")
    m3.metric("Grid Price", f"${price:.2f}/kWh")

    # Forecast Trend
    st.subheader("📈 24-Hour Predictive Forecast")
    trend_fig = px.line(df, x=df.index, y=available_apps)
    st.plotly_chart(trend_fig, use_container_width=True)
