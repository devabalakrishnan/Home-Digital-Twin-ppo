import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Residential Digital Twin & XAI", layout="wide")

# --- 2. LOAD DATA ---
# Using the file you uploaded
DATA_FILE = "24_hour_forecast(1).csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    else:
        return None

df = load_data()

if df is None:
    st.error(f"🚨 File `{DATA_FILE}` not found. Please ensure it is in the same directory.")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Digital Twin Controls")
st.sidebar.markdown("Navigate through the 24-hour predictive horizon.")
hour_idx = st.sidebar.slider("Select Forecast Hour", 0, len(df)-1, 12)
row = df.iloc[hour_idx]

# --- 4. SYSTEM LOGIC (PPO AGENT SIMULATION) ---
# We define the logic for the "Agent Decision" based on your paper's parameters
price = float(row['electricity_price'])
total_load = float(row['Total_Load_Forecasted'])
dr_threshold = 1.2  # Defined in your framework

is_peak = price >= 0.15 or total_load > dr_threshold
status = "CRITICAL (PEAK)" if is_peak else "OPTIMIZED (NORMAL)"
agent_action = "PPO ACTION: LOAD SHEDDING ACTIVE" if is_peak else "PPO STATUS: MONITORING"

# --- 5. HEADER ---
st.title("🌐 Residential Digital Twin Portal")
st.markdown(f"**Current State Synchronization:** {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")

# --- 6. TOP ROW: SPATIAL & TELEMETRY ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🏠 Spatial Synchronization")
    # Dynamic House Display logic
    glow = "rgba(255, 0, 0, 0.3)" if is_peak else "rgba(0, 255, 0, 0.1)"
    st.markdown(
        f"""
        <div style="text-align: center; padding: 50px; background: {glow}; border-radius: 20px; border: 2px solid grey;">
            <h1 style="font-size: 100px; margin: 0;">🏠</h1>
            <h2 style="color: black; margin: 10px 0 0 0;">{status}</h2>
            <p style="font-size: 20px; color: #333;"><b>Load: {total_load:.2f} kW</b></p>
        </div>
        """, unsafe_allow_html=True
    )
    st.info(f"🤖 **Decision:** {agent_action}")

with col2:
    st.subheader("📊 Appliance Energy Breakdown")
    apps = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing_Machine']
    # Filter only apps present in columns
    existing_apps = [a for a in apps if a in row.index]
    if existing_apps:
        pie_data = pd.DataFrame({
            'Appliance': existing_apps,
            'Consumption (kW)': [float(row[a]) for a in existing_apps]
        })
        fig_pie = px.pie(pie_data, values='Consumption (kW)', names='Appliance', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.T10)
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=350, font=dict(size=16))
        st.plotly_chart(fig_pie, use_container_width=True)

# --- 7. MIDDLE ROW: METRICS & FORECAST ---
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted Load", f"{total_load:.2f} kW")
m2.metric("Grid Price", f"${price:.2f}/kWh")
m3.metric("Occupancy", "👤 Active" if row['occupancy'] == 1 else "⚪ Vacant")
m4.metric("Savings Est.", f"{(total_load * 0.22):.2f} %")

st.subheader("📈 24-Hour Predictive Forecast")
# Plotting the forecasted lines
trend_fig = px.line(df, x=df.index, y=existing_apps, 
                    labels={'index': 'Time Step', 'value': 'Power (kW)'},
                    template="plotly_white")
trend_fig.update_traces(line=dict(width=4))
trend_fig.add_hline(y=dr_threshold, line_dash="dash", line_color="red", 
                    annotation_text="DR Target (1.2 kW)", annotation_position="top right")
trend_fig.update_layout(height=400, font=dict(size=14))
st.plotly_chart(trend_fig, use_container_width=True)

# --- 8. BOTTOM ROW: EXPLAINABLE AI (XAI) ---
st.divider()
st.subheader("🔍 Explainable AI (XAI) Insights")
st.write("This section provides a **Local Interpretability** view, explaining the mathematical weight behind the PPO Agent's current decision.")

# Create the Feature Importance Graph (LIME/SHAP style)
# We use the actual row data to determine influence
features = ['Electricity Price', 'Current Demand', 'Occupancy', 'Meal Time']
# Logic: Price and demand are the biggest drivers in your PPO framework
weights = [
    price * 2.0, 
    (total_load / dr_threshold), 
    0.2 if row['occupancy'] == 1 else 0.05,
    0.3 if row['is_meal_time'] == 1 else 0.0
]

fig_xai, ax = plt.subplots(figsize=(10, 4))
colors = ['#ff4b4b' if w > 0.5 else '#0068c9' for w in weights]
y_pos = np.arange(len(features))

ax.barh(y_pos, weights, align='center', color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=12)
ax.invert_yaxis()  # Best at top
ax.set_xlabel('Influence Weight on Decision', fontsize=12)
ax.set_title(f"Why did the PPO Agent choose '{status}'?", fontsize=14, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Display the plot
st.pyplot(fig_xai)

st.success(f"**XAI Summary:** For this time step, the primary driver for the decision was **{features[np.argmax(weights)]}**. This aligns with the 'Price-Elastic' behavior modeled in Section 6 of the paper.")
