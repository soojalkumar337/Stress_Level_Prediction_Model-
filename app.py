import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title='Stress Level Prediction', layout='centered')
st.title("🧠 Stress Level Prediction")

st.write("""Use the sliders and inputs to enter features. The model will predict a stress category: Low, Medium, or High.""")

# Load artifacts
try:
    model = joblib.load('stress_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
except Exception as e:
    st.error('Model files not found. Run the training notebook/script first to create stress_model.pkl, scaler.pkl, and label_encoder.pkl.')
    st.stop()

# Input widgets
age = st.slider('Age', 18, 80, 30)
bmi = st.slider('BMI', 15.0, 40.0, 25.0)
sleep_hours = st.slider('Sleep hours (per night)', 3.0, 10.0, 7.0)
heart_rate = st.slider('Average resting heart rate', 40.0, 120.0, 70.0)
activity_level = st.slider('Activity level (0-100)', 0, 100, 50)
caffeine_mg = st.slider('Daily caffeine (mg)', 0.0, 600.0, 150.0)
work_hours = st.slider('Work hours per day', 0.0, 16.0, 8.0)
meditation_min = st.slider('Meditation minutes per day', 0.0, 60.0, 5.0)
screen_time = st.slider('Daily screen time (hours)', 0.0, 16.0, 6.0)
social_support = st.slider('Social support (0-5)', 0, 5, 3)

input_df = pd.DataFrame([{
    'age': age,
    'bmi': bmi,
    'sleep_hours': sleep_hours,
    'heart_rate': heart_rate,
    'activity_level': activity_level,
    'caffeine_mg': caffeine_mg,
    'work_hours': work_hours,
    'meditation_min': meditation_min,
    'screen_time_hours': screen_time,
    'social_support': social_support
}])

X_scaled = scaler.transform(input_df)
pred = model.predict(X_scaled)[0]
pred_label = le.inverse_transform([pred])[0]
prob = model.predict_proba(X_scaled).max()

st.markdown(f"### Predicted stress level: **{pred_label}**")
st.markdown(f"**Confidence:** {prob*100:.2f}%")

st.write('---')
st.write('Feature values:')
st.write(input_df.T)
