import streamlit as pd_st 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib


st.set_page_config(page_title="OptiProd AI Portal", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def load_model():
    return joblib.load('predictive_model.pkl')

model = load_model()


st.title("🏭 OptiProd AI")
st.subheader("Digital Production Management & Predictive Maintenance Portal")
st.markdown("---")

st.sidebar.header("🛠️ Live Telemetry Simulator")
st.sidebar.markdown("Adjust the sliders below to simulate live machine sensor feeds.")


temp_input = st.sidebar.slider("Machine Temperature (°C)", 40.0, 110.0, 72.0, step=0.5)
vib_input = st.sidebar.slider("Vibration Amplitude (mm/s)", 1.0, 10.0, 4.2, step=0.1)
wear_input = st.sidebar.slider("Tool Usage Duration (Hours)", 0, 120, 45)



input_data = pd.DataFrame([[temp_input, vib_input, wear_input]], 
                          columns=['Temperature', 'Vibration', 'Tool_Wear_Hours'])

risk_probability = model.predict_proba(input_data)[0][1] * 100

# --- MAIN DASHBOARD VISUALS ---


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Current Production Line Status", value="OPERATIONAL", delta="Optimal Speed")

with col2:
    st.metric(label="Active IoT Sensors connected", value="3 / 3 Node active")

with col3:
   
    if risk_probability < 30:
        st.success(f"Failure Risk: {risk_probability:.1f}% (Healthy)")
    elif risk_probability < 70:
        st.warning(f"Failure Risk: {risk_probability:.1f}% (Caution)")
    else:
        st.error(f"CRITICAL RISK: {risk_probability:.1f}% (Maintenance Required!)")

st.markdown("### Real-time Sensor Visualizations")


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = risk_probability,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "AI Calculated Breakdown Risk Probability (%)"},
    gauge = {
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 40], 'color': "lightgreen"},
            {'range': [40, 70], 'color': "orange"},
            {'range': [70, 100], 'color': "red"}
        ],
    }
))

fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)


if risk_probability >= 70:
    st.error("⚠️ **Automated Action Protocol:** AI recommends scheduling an immediate 15-minute diagnostic window to avoid tool snapping.")