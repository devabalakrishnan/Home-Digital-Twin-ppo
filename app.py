import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import plotly.express as px

# --- 1. SETTINGS & PATHS ---
st.set_page_config(page_title="Home Digital Twin AI", layout="wide")

# Paths for the data and assets
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
    val = float(load_value)
    
    # Define colors for different states
    glows = {
        "CRITICAL (PEAK)": "rgba(255, 0, 0, 0.4)", 
        "OPTIMIZED (OFF-PEAK)": "rgba(0, 255, 0, 0.2)"
    }
    glow_color = glows.get(str(load_status), "rgba(0, 255, 0, 0.2)")

    img_base64 = get_base64_of_bin_file(ASSET_PATH)
    
    if img_base64:
        # Use uploaded SVG asset
        img_html = f'<img src="data:image/svg+xml;base64,{img_base64}" width="100%" style="max-width: 350px;">'
    else:
        # Fallback to high-quality icon if SVG is missing
        url = "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/house-chimney.svg"
        img_html = f'<img src="{url}" width="120" style="filter: invert(0.5); opacity: 0.8;">'

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

# Load prediction data
if not os.path.exists(DATA_PATH):
    st.error(f"🚨 Data file not found at `{DATA_PATH}`. Please run your training script first.")
else:
    df = pd.read_csv(DATA_PATH)
    
    # Sidebar Control: Navigate through the 24-hour forecast
    st.sidebar.header("🕹️ Controls")
    hour_idx = st.sidebar.slider("Select Forecast Hour", 0, len(df)-1, 0)
    row = df.iloc[hour_idx]

    # --- 4. DATA EXTRACTION & PPO LOGIC ---
    price = float(row.get('electricity_price', 0))
    
    # Identify available appliances in the dataset
    appliances = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing Machine']
    available_apps = [app for app in appliances if app in row.index]
    
    # Calculate initial total load
    total_raw_load = float(row[available_apps].sum())

    # BINARY PPO LOGIC: Decision based on cost (Peak vs. Off-Peak)
    # Using 0.5 as threshold to match the provided synthetic dataset
    if price >= 0.5: 
        status = "CRITICAL (PEAK)"
        ppo_msg = "PPO ACTION: PEAK LOAD SHEDDING"
        
        # Calculate reduction: 70% of heavy appliances (Heater and Washing Machine)
        heater_val = float(row.get('Heater', 0))
        wm_val = float(row.get('Washing Machine', 0))
        reduction = (heater_val * 0.7) + (wm_val * 0.7)
        
        optimized_load = total_raw_load - reduction
    else: 
        status = "OPTIMIZED (OFF-PEAK)"
        ppo_msg = "PPO STATUS: NORMAL OPERATION"
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
            # Current hour appliance distribution
            pie_values = [float(row[app]) for app in available_apps]
            fig = px.pie(values=pie_values, names=available_apps, hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No appliance data available for this hour.")

    st.divider()

    # Efficiency Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Load", f"{total_raw_load:.2f} kW")
    m2.metric("Optimized Load", f"{float(optimized_load):.2f} kW", f"{float(optimized_load - total_raw_load):.2f} kW")
    m3.metric("Grid Price", f"${price:.2f}/kWh")

    # --- 6. PREDICTIVE TREND GRAPH ---
    st.subheader("📈 24-Hour Predictive Forecast")
    
    # Creating a line chart for the 24-hour horizon
    trend_fig = px.line(df, x=df.index, y=available_apps, 
                        labels={'index': 'Hour', 'value': 'Power (kW)'},
                        template="plotly_white")
    
    # THICK LINES: Set line width to 4 for high visibility
    trend_fig.update_traces(line=dict(width=4)) 
    
    # Clean up layout for presentation slides
    trend_fig.update_layout(
        hovermode="x unified", 
        legend_title_text='Appliances',
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    st.plotly_chart(trend_fig, use_container_width=True)
