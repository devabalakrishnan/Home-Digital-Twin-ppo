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

# --- 2. HOUSE VISUALIZATION FUNCTIONS ---
def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return None
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def display_custom_house(load_status, load_value):
    if load_status == "CRITICAL":
        glow_color = "rgba(255, 0, 0, 0.5)"
    elif load_status == "HIGH":
        glow_color = "rgba(255, 165, 0, 0.5)"
    else:
        glow_color = "rgba(0, 255, 0, 0.3)"

    img_base64 = get_base64_of_bin_file(ASSET_PATH)
    
    if img_base64:
        st.markdown(
            f"""
            <div style="position: relative; text-align: center; padding: 20px; 
                        background: {glow_color}; border-radius: 20px; transition: 0.5s;">
                <img src="data:image/svg+xml;base64,{img_base64}" width="100%" style="max-width: 400px;">
                <div style="margin-top: 10px;">
                    <h3 style="margin:0;">Live Load: {load_value:.2f} kW</h3>
                    <small>State: {load_status}</small>
                </div>
            </div>
            """,
            unsafe_allow_code=True
        )
    else:
        st.error(f"Asset not found at {ASSET_PATH}. Please check your assets folder.")

# --- 3. MAIN APP LOGIC ---
st.title("🌐 Residential Digital Twin: XGBoost + PPO")

if not os.path.exists(DATA_PATH):
    st.warning("Prediction data not found. Please run 'python src/train_models.py' first.")
else:
    df = pd.read_csv(DATA_PATH)
    
    # Sidebar Control
    st.sidebar.header("Simulation Control")
    hour = st.sidebar.slider("Select Forecast Hour", 0, 23, 12)
    current_data = df.iloc[hour]

    # PPO Logic (Simulated from your ppo_agent.py)
    price = current_data['electricity_price']
    occupancy = current_data['occupancy']
    
    # Identify Load State
    if price > 0.6 and occupancy < 3:
        status = "CRITICAL"
        # Simulate PPO Action: Reducing Load
        optimized_load = (current_data['Fridge'] + current_data['Heater'] * 0.4) 
    elif price > 0.4:
        status = "HIGH"
        optimized_load = (current_data['Fridge'] + current_data['Heater'])
    else:
        status = "OPTIMIZED"
        optimized_load = (current_data['Fridge'] + current_data['Heater'])

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Spatial Twin View")
        display_custom_house(status, optimized_load)

    with col2:
        st.subheader("Predictive Analytics (XGBoost)")
        # Show appliance breakdown
        appliances = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
        fig = px.pie(values=current_data[appliances], names=appliances, title="Appliance Energy Stake")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("24-Hour Optimization Trend")
    st.line_chart(df.set_index('datetime')[['electricity_price', 'occupancy']])

