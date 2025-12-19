 import streamlit as st
import base64
from PIL import Image

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def display_custom_house(load_status, load_value):
    # Determine the glow color based on status
    if load_status == "CRITICAL":
        glow_color = "rgba(255, 0, 0, 0.5)" # Red glow
    elif load_status == "HIGH":
        glow_color = "rgba(255, 165, 0, 0.5)" # Orange glow
    else:
        glow_color = "rgba(0, 255, 0, 0.3)" # Green glow

    # Path to your uploaded asset
    img_path = "assets/house_model.svg" 
    img_base64 = get_base64_of_bin_file(img_path)

    # HTML/CSS to display your image with a dynamic glow effect
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

# Inside your main loop, call it like this:
# display_custom_house("OPTIMIZED", 1.24)