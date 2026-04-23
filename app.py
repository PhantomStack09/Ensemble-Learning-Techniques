import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Weather Predictor", page_icon="🌤️", layout="centered")

# Custom CSS for premium design
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
    }
    h1 {
        color: #60a5fa;
        font-family: 'Inter', sans-serif;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .prediction-box {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        text-align: center;
        margin-top: 20px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌤️ AI Weather Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Ensemble Learning powered weather forecasting</p>", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, scaler, le
    except FileNotFoundError:
        return None, None, None

model, scaler, le = load_models()

if model is None:
    st.warning("Model files not found. Please train the model in the Jupyter Notebook first.")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
        humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
        
    with col2:
        wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        pressure = st.slider("Pressure (hPa)", min_value=980.0, max_value=1050.0, value=1013.0, step=1.0)
        
    if st.button("Predict Weather Condition"):
        # Prepare input
        input_data = pd.DataFrame({
            'Temperature_C': [temperature],
            'Humidity_%': [humidity],
            'Wind_Speed_kmh': [wind_speed],
            'Pressure_hPa': [pressure]
        })
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction_encoded = model.predict(input_scaled)[0]
        prediction = le.inverse_transform([prediction_encoded])[0]
        
        # Icons based on prediction
        icon = "☀️"
        color = "#F59E0B"
        if prediction == "Rainy":
            icon = "🌧️"
            color = "#3B82F6"
        elif prediction == "Cloudy":
            icon = "☁️"
            color = "#6B7280"
            
        st.markdown(f"""
            <div class='prediction-box'>
                <h3 style='color: #f1f5f9; margin-bottom: 10px;'>Prediction Result</h3>
                <div style='font-size: 48px;'>{icon}</div>
                <h2 style='color: {color}; margin-top: 10px;'>{prediction}</h2>
            </div>
        """, unsafe_allow_html=True)
