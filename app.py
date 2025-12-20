import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import plotly.express as px

# --- 1. SETTINGS & PATHS ---
st.set_page_config(page_title="Home Digital Twin AI", layout="wide")

# Correct paths for GitHub deployment
DATA_PATH = os.path.join('data', 'next_day_prediction.csv')
ASSET_PATH = os.path.join('assets', 'house_model.svg')

# --- 2. ASSET & VISUALIZATION FUNCTIONS ---
def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return None
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

def display_custom_house(load_status, load_value):
    # CRITICAL: Force load_value to be a standard Python float
    try:
        val = float(load_value)
    except:
        val = 0.0
    
    glows = {
        "CRITICAL": "rgba(255, 0, 0, 0.4)", 
        "HIGH": "rgba(255, 165, 0, 0.4)", 
        "OPTIMIZED": "rgba(0, 255, 0, 0.2)"
    }
    glow_color = glows.get(str(load_status), glows["OPTIMIZED"])

    img_base64 = get_base64_of_bin_file(ASSET_PATH)
    
    if img_base64:
        img_html = f'<img src="data:image/svg+xml;base64,{img_base64}" width="100%" style="max-width: 350px;">'
    else:
        # High-quality fallback icon
        url = "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/house-chimney.svg"
        img_html = f'<img src="{url}" width="120" style="filter: invert(0.5); opacity: 0.8;">'

    # The string below is where your error was occurring. 
    # Ensuring 'val' is a float and 'glow_color' is a string fixes it.
    st.markdown(
        f"""
        <div style="text-align: center; padding: 40px; background: {str(glow_color)}; border-radius: 25px; border: 2px solid rgba(255,255,255,0.1);">
            {img_html}
            <div style="margin-top: 20px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 10px;">
                <h3 style="margin:0; color: black;">Live Load: {val:.2f} kW</h3>
                <p style="margin:0; color: #333;">State: <b>{str(load_status)}</b></p>
            </div>
        </div>
        """,
        unsafe_allow_code=True
    )

# --- 3. MAIN APP LOGIC ---
st.title("🌐 Residential Digital Twin")

if not os.path.exists(DATA_PATH):
    st.error(f"🚨 Data file not found. Ensure '{DATA_PATH}' exists.")
else:
    df = pd.read_csv(DATA_PATH)
    
    st.sidebar.header("🕹️ Simulation Control")
    hour_idx = st.sidebar.slider("Select Forecast Hour", 0, len(df)-1, 0)
    
    # Extracting current row
    row = df.iloc[hour_idx]

    # --- 4. DATA EXTRACTION & PPO LOGIC ---
    # Convert everything to float() immediately to prevent TypeErrors
    price = float(row.get('electricity_price', 0))
    occupancy = float(row.get('occupancy', 0))
    
    # List of appliances to check
    appliances = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing Machine']
    available_apps = [app for app in appliances if app in row.index]
    
    # Calculate sum and force to float
    total_raw_load = float(row[available_apps].sum())

    # PPO Agent Decision logic
    if price > 0.6:
        status = "CRITICAL"
        # Reduce Heater/Washing Machine by 70% if they exist
        reduction = (float(row.get('Heater', 0)) * 0.7) + (float(row.get('Washing Machine', 0)) * 0.7)
        optimized_load = total_raw_load - reduction
        ppo_msg = "PPO SHEDDING LOAD"
    elif price > 0.4:
        status = "HIGH"
        optimized_load = total_raw_load
        ppo_msg = "PPO MONITORING"
    else:
        status = "OPTIMIZED"
        optimized_load = total_raw_load
        ppo_msg = "NORMAL OPERATION"

    # --- 5. DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("🏠 Spatial Synchronization")
        display_custom_house(status, optimized_load)
        st.info(f"**Agent Decision:** {ppo_msg}")

    with col2:
        st.subheader("📊 Energy Telemetry")
        if available_apps:
            # Prepare data for pie chart
            pie_values = [float(row[app]) for app in available_apps]
            fig = px.pie(values=pie_values, names=available_apps, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    # KPI Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Load", f"{total_raw_load:.2f} kW")
    m2.metric("Optimized Load", f"{float(optimized_load):.2f} kW", f"{float(optimized_load - total_raw_load):.2f} kW")
    m3.metric("Current Price", f"${price:.2f}/kWh")
