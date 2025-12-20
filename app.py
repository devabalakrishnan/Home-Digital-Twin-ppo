import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import plotly.express as px

# --- 1. SETTINGS & PATHS ---
st.set_page_config(page_title="Home Digital Twin AI", layout="wide")

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
    # Ensure load_value is a float to prevent TypeError
    val = float(load_value)
    
    glows = {
        "CRITICAL": "rgba(255, 0, 0, 0.4)", 
        "HIGH": "rgba(255, 165, 0, 0.4)", 
        "OPTIMIZED": "rgba(0, 255, 0, 0.2)"
    }
    glow_color = glows.get(load_status, glows["OPTIMIZED"])

    img_base64 = get_base64_of_bin_file(ASSET_PATH)
    
    if img_base64:
        img_html = f'<img src="data:image/svg+xml;base64,{img_base64}" width="100%" style="max-width: 350px;">'
    else:
        url = "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/house-chimney.svg"
        img_html = f'<img src="{url}" width="120" style="filter: invert(0.5); opacity: 0.8;">'

    # The f-string now receives a clean float (val)
    st.markdown(
        f"""
        <div style="text-align: center; padding: 40px; background: {glow_color}; border-radius: 25px; border: 2px solid rgba(255,255,255,0.1);">
            {img_html}
            <div style="margin-top: 20px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 10px;">
                <h3 style="margin:0; color: black;">Live Load: {val:.2f} kW</h3>
                <p style="margin:0; color: #333;">State: <b>{load_status}</b></p>
            </div>
        </div>
        """,
        unsafe_allow_code=True
    )

# --- 3. MAIN APP LOGIC ---
st.title("🌐 Residential Digital Twin")

if not os.path.exists(DATA_PATH):
    st.error(f"🚨 Data file not found at `{DATA_PATH}`. Run `train_models.py` first!")
else:
    df = pd.read_csv(DATA_PATH)
    
    st.sidebar.header("🕹️ Simulation Control")
    hour = st.sidebar.slider("Select Forecast Hour", 0, len(df)-1, 0)
    
    # Extracting current row
    current_data = df.iloc[hour]

    # --- 4. DATA EXTRACTION & PPO LOGIC ---
    # Convert everything to float() to avoid TypeErrors
    price = float(current_data['electricity_price'])
    occupancy = float(current_data['occupancy'])
    
    appliances = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing Machine']
    available_apps = [app for app in appliances if app in current_data.index]
    
    base_load = float(current_data[available_apps].sum())

    # Decision Logic
    if price > 0.6:
        status = "CRITICAL"
        ppo_action = "PPO SHEDDING LOAD"
        optimized_load = base_load - (float(current_data.get('Heater', 0)) * 0.6)
    elif price > 0.4:
        status = "HIGH"
        ppo_action = "PPO MONITORING"
        optimized_load = base_load
    else:
        status = "OPTIMIZED"
        ppo_action = "NORMAL OPERATION"
        optimized_load = base_load

    # --- 5. DASHBOARD LAYOUT ---
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("🏠 Spatial Synchronization")
        # Passing clean variables to the function
        display_custom_house(status, optimized_load)
        st.info(f"**Agent Decision:** {ppo_action}")

    with col2:
        st.subheader("📊 Energy Telemetry")
        if available_apps:
            fig = px.pie(values=current_data[available_apps], names=available_apps, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Unoptimized", f"{base_load:.2f} kW")
    c2.metric("Optimized", f"{float(optimized_load):.2f} kW")
    c3.metric("Grid Price", f"${price:.2f}/kWh")
