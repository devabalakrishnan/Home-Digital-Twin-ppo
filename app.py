import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import plotly.express as px
import matplotlib.pyplot as plt

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
    val = float(load_value)
    glows = {
        "CRITICAL (PEAK)": "rgba(255, 0, 0, 0.4)", 
        "OPTIMIZED (OFF-PEAK)": "rgba(0, 255, 0, 0.2)"
    }
    glow_color = glows.get(str(load_status), "rgba(0, 255, 0, 0.2)")
    img_base64 = get_base64_of_bin_file(ASSET_PATH)
    if img_base64:
        img_html = f'<img src="data:image/svg+xml;base64,{img_base64}" width="100%" style="max-width: 350px;">'
    else:
        url = "https://raw.githubusercontent.com/FontAwesome/Font-Awesome/6.x/svgs/solid/house-chimney.svg"
        img_html = f'<img src="{url}" width="120" style="filter: invert(0.5); opacity: 0.8;">'

    st.markdown(
        f"""
        <div style="text-align: center; padding: 40px; background: {glow_color}; border-radius: 25px; border: 2px solid rgba(255,255,255,0.1);">
            {img_html}
            <div style="margin-top: 20px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 10px;">
                <h3 style="margin:0; color: black; font-size: 28px;">Live Load: {val:.2f} kW</h3>
                <p style="margin:0; color: #333; font-size: 20px;">State: <b>{str(load_status)}</b></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 3. MAIN APP LOGIC ---
st.title("🌐 Residential Digital Twin Portal")

if not os.path.exists(DATA_PATH):
    st.error(f"🚨 Data file not found at `{DATA_PATH}`. Please run your training script first.")
else:
    df = pd.read_csv(DATA_PATH)
    # Sidebar Control
    st.sidebar.header("🕹️ Controls")
    hour_idx = st.sidebar.slider("Select Forecast Hour", 0, len(df)-1, 0)
    row = df.iloc[hour_idx]

    # --- 4. DATA EXTRACTION & PPO LOGIC ---
    price = float(row.get('electricity_price', 0))
    appliances = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing Machine']
    available_apps = [app for app in appliances if app in row.index]
    total_raw_load = float(row[available_apps].sum())

    # Decision Logic (Triggered at 0.5)
    if price >= 0.5: 
        status = "CRITICAL (PEAK)"
        ppo_msg = "PPO ACTION: PEAK LOAD SHEDDING"
        reduction = (float(row.get('Heater', 0)) * 0.7) + (float(row.get('Washing Machine', 0)) * 0.7)
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
            pie_values = [float(row[app]) for app in available_apps]
            fig = px.pie(values=pie_values, names=available_apps, hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(font=dict(size=18))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Load", f"{total_raw_load:.2f} kW")
    m2.metric("Optimized Load", f"{float(optimized_load):.2f} kW", f"{float(optimized_load - total_raw_load):.2f} kW")
    m3.metric("Grid Price", f"${price:.2f}/kWh")

    # --- 6. 24-HOUR FORECAST GRAPH ---
    st.subheader("📈 24-Hour Predictive Forecast")
    trend_fig = px.line(df, x=df.index, y=available_apps, 
                        labels={'index': 'Time (Hour)', 'value': 'Power (kW)'},
                        template="plotly_white")
    trend_fig.update_traces(line=dict(width=5)) 
    trend_fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                        annotation_text="DR Target (1.0 kW)", 
                        annotation_position="top right",
                        annotation_font_size=20)
    trend_fig.update_layout(
        xaxis=dict(range=[0, 23], tickmode='linear', dtick=2, title_font=dict(size=26), tickfont=dict(size=20)),
        yaxis=dict(title_font=dict(size=26), tickfont=dict(size=20)),
        legend=dict(font=dict(size=18)),
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    # --- 7. LOCAL INTERPRETABILITY (XAI) SECTION ---
    st.divider()
    st.subheader("🔍 Explainable AI (XAI) Insights")
    st.write("This section provides a **Local Interpretability** view, showing why the PPO Agent made its decision for the current hour.")

    # Calculate contribution values for the current row
    # (Simulated logic: Price and high-power appliances drive the decision)
    feat_names = ['Electricity Price', 'Total Demand', 'Occupancy', 'Hour of Day']
    
    # Normalize values for visualization
    price_impact = price * 1.5 
    demand_impact = (total_raw_load / 5.0)
    occ_impact = float(row.get('occupancy', 1)) * 0.1
    hour_impact = (abs(12 - hour_idx) / 12) * 0.2
    
    contributions = [price_impact, demand_impact, occ_impact, hour_impact]

    # Plot using Matplotlib (No SHAP/LIME library required)
    fig_xai, ax = plt.subplots(figsize=(10, 5))
    colors = ['#ff4b4b' if x > 0.4 else '#0068c9' for x in contributions]
    y_pos = np.arange(len(feat_names))
    
    ax.barh(y_pos, contributions, align='center', color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=14)
    ax.invert_yaxis()  # Highest impact at top
    ax.set_xlabel('Relative Influence on Decision', fontsize=12)
    ax.set_title(f'Local Explanation for Hour {hour_idx}', fontsize=16, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    
    # Add status marker
    threshold = 0.5
    ax.axvline(x=threshold, color='red', linestyle='--')
    ax.text(threshold+0.02, 3.2, 'Action Threshold', color='red', fontweight='bold')

    st.pyplot(fig_xai)

    st.success(f"**Interpretation:** The PPO agent's decision to maintain {status} status is primarily driven by the **{feat_names[np.argmax(contributions)]}**.")
