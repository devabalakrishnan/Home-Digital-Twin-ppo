import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import plotly.express as px

# --- 1. SETTINGS & PATHS ---
st.set_page_config(page_title="Home Digital Twin AI", layout="wide")

# Ensure paths are relative and correct for GitHub/Streamlit Cloud
DATA_PATH = os.path.join('data', 'next_day_prediction.csv')
ASSET_PATH = os.path.join('assets', 'house_model.svg')

# --- 2. ASSET & VISUALIZATION FUNCTIONS ---
def get_base64_of_bin_file(bin_file):
    """Loads a local file and converts to base64 for HTML display."""
    if not os.path.exists(bin_file):
        return None
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

def display_custom_house(load_status, load_value):
    """Displays the house model with a dynamic glow based on energy state."""
    glows = {
        "CRITICAL": "rgba(255, 0, 0, 0.4)", 
        "HIGH": "rgba(255, 165, 0, 0.4)", 
        "OPTIMIZED": "rgba(0, 255, 0, 0.2)"
    }
    glow_color = glows.get(load_status, glows["OPTIMIZED"])

    img_base64 = get_base64_of_bin_file(ASSET_PATH)
    
    if img_base64:
        # If your local SVG exists
        img_html = f'<img src="data:image/svg+xml;base64,{img_base64}" width="100%" style="max-width: 350px;">'
    else:
        # FALLBACK: A high-quality house icon from the web if your file is missing
        url = "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/house-chimney.svg"
        img_html = f'<img src="{url}" width="120" style="filter: invert(0.5); opacity: 0.8;">'

    st.markdown(
        f"""
        <div style="text-align: center; padding: 40px; background: {glow_color}; border-radius: 25px; border: 2px solid rgba(255,255,255,0.1); transition: 0.5s;">
            {img_html}
            <div style="margin-top: 20px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 10px;">
                <h3 style="margin:0; color: black;">Live Load: {load_value:.2f} kW</h3>
                <p style="margin:0; color: #333;">State: <b>{load_status}</b></p>
            </div>
        </div>
        """,
        unsafe_allow_code=True
    )

# --- 3. MAIN APP LOGIC ---
st.title("🌐 Residential Digital Twin")
st.markdown("### XGBoost Forecasting & PPO Optimization Engine")

if not os.path.exists(DATA_PATH):
    st.error(f"🚨 **Data Error:** File not found at `{DATA_PATH}`. Please run `python src/train_models.py` first to generate the forecast.")
else:
    df = pd.read_csv(DATA_PATH)
    
    # Sidebar Control
    st.sidebar.header("🕹️ Simulation Control")
    hour = st.sidebar.slider("Select Forecast Hour (24h)", 0, len(df)-1, 0)
    current_data = df.iloc[hour]

    # --- 4. PPO OPTIMIZATION LOGIC ---
    price = current_data['electricity_price']
    occupancy = current_data['occupancy']
    
    # Define appliances precisely to match your CSV
    appliances = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing Machine']
    # Defensive check: Only use columns that actually exist in the CSV
    available_apps = [app for app in appliances if app in current_data.index]
    
    base_load = current_data[available_apps].sum()

    # Simulated PPO Agent Decision logic
    if price > 0.6:
        status = "CRITICAL"
        ppo_action = "PPO SHEDDING LOAD"
        # Simulate RL action: Reduce Heater and Washing Machine by 60%
        optimized_load = base_load
        if 'Heater' in current_data: optimized_load -= (current_data['Heater'] * 0.6)
        if 'Washing Machine' in current_data: optimized_load -= (current_data['Washing Machine'] * 0.6)
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
        display_custom_house(status, optimized_load)
        
        st.info(f"**Agent Decision:** {ppo_action}")
        st.metric("Grid Price", f"${price}/kWh", delta="High Tariff" if price > 0.5 else "Standard")

    with col2:
        st.subheader("📊 Energy Telemetry")
        if available_apps:
            # Fixing the KeyError by using only available_apps
            fig = px.pie(
                values=current_data[available_apps], 
                names=available_apps, 
                title="Predicted Appliance Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No appliance data columns found in the CSV.")

    st.divider()
    
    # Performance Metrics
    c1, c2, c3 = st.columns(3)
    savings = (base_load - optimized_load) * price
    c1.metric("Unoptimized Load", f"{base_load:.2f} kW")
    c2.metric("PPO Optimized", f"{optimized_load:.2f} kW", f"-{((base_load-optimized_load)/base_load)*100:.1f}%")
    c3.metric("Cost Savings", f"${savings:.4f}")

    # Forecast Chart
    st.subheader("📈 24-Hour Predictive Trend (XGBoost Output)")
    trend_fig = px.line(df, x=df.index, y=available_apps, title="Next-Day Load Forecast per Appliance")
    st.plotly_chart(trend_fig, use_container_width=True)

