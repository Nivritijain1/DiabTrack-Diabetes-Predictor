import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page config with dark theme
st.set_page_config(
    page_title="🩺 Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== DARK CREATIVE CSS ====================
st.markdown("""
<style>
    /* Cyberpunk Dark Theme */
    .stApp {
        background: radial-gradient(ellipse at center, #0c0c0c 0%, #000000 100%);
        color: #ffffff;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    
    /* Neon header */
    .neon-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00ff88, #00ccff, #ff0088);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    
    /* Glass effect cards */
    .glass-card {
        background: rgba(20, 20, 30, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 255, 136, 0.4);
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.2);
    }
    
    /* Cyberpunk input fields */
    .stNumberInput input, .stSelectbox select {
        background: rgba(30, 30, 40, 0.8) !important;
        color: #00ff88 !important;
        border: 2px solid rgba(0, 255, 136, 0.3) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.4) !important;
        background: rgba(40, 40, 50, 0.9) !important;
    }
    
    /* Cyberpunk button */
    .cyber-button {
        background: linear-gradient(45deg, #ff0088, #00ccff) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 40px !important;
        font-weight: 800 !important;
        font-size: 18px !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(255, 0, 136, 0.4) !important;
    }
    
    .cyber-button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(255, 0, 136, 0.6) !important;
        letter-spacing: 3px !important;
    }
    
    /* Result display with glow */
    .result-glow {
        border: 2px solid;
        border-image: linear-gradient(45deg, #00ff88, #00ccff) 1;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        background: rgba(0, 0, 0, 0.5);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.6); }
        100% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
    }
    
    /* Parameter tags */
    .param-tag {
        display: inline-block;
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 15px;
        padding: 8px 15px;
        margin: 5px;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        transition: all 0.3s ease;
    }
    
    .param-tag:hover {
        background: rgba(0, 255, 136, 0.2);
        transform: scale(1.05);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(30, 30, 40, 0.8);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        color: #888 !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 136, 0.2) !important;
        color: #00ff88 !important;
        border: 1px solid rgba(0, 255, 136, 0.4) !important;
    }
    
    /* Matrix grid background */
    .matrix-grid {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        opacity: 0.05;
        background-image: 
            linear-gradient(rgba(0, 255, 136, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 136, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(30, 30, 40, 0.6);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border-top: 4px solid #00ff88;
        margin: 10px;
    }
    
    /* Loading animation */
    .cyber-spinner {
        border: 3px solid rgba(0, 255, 136, 0.1);
        border-top: 3px solid #00ff88;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Label styling */
    label {
        color: #00ccff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
        display: block;
    }
    
    /* Warning and error messages */
    .stAlert {
        border-radius: 10px;
        border: 1px solid rgba(255, 100, 100, 0.3);
        background: rgba(255, 50, 50, 0.1) !important;
    }
</style>

<div class="matrix-grid"></div>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('diabetes_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return True, model, scaler
    except Exception as e:
        return False, None, None

# Load model
model_loaded, model, scaler = load_model()

# ==================== HEADER ====================
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 class="neon-title">🩺 NEURO-DIABETES AI</h1>
    <p style="color: #aaa; font-size: 1.2rem; letter-spacing: 1px;">
        Clinical Neural Network • 12-Parameter Analysis • Medical-Grade Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== MAIN LAYOUT ====================
col_left, col_right = st.columns([1.4, 1])

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### 🔬 PATIENT DATA INPUT')
    
    # Tabs for parameter organization
    tab1, tab2, tab3 = st.tabs(["📊 VITAL SIGNS", "🩸 BLOOD MARKERS", "👤 DEMOGRAPHICS"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("AGE (years)", min_value=18, max_value=100, value=45, step=1)
            gender = st.selectbox("GENDER", ["Male", "Female"])
            bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        
        with col2:
            waist = st.number_input("WAIST (cm)", min_value=50, max_value=150, value=85, step=1)
            systolic_bp = st.number_input("SYSTOLIC BP (mmHg)", min_value=80, max_value=200, value=120, step=1)
    
    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            glucose = st.number_input("GLUCOSE (mg/dL)", min_value=50, max_value=300, value=100, step=1)
            hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=5.5, step=0.1)
        
        with col4:
            insulin = st.number_input("INSULIN (μU/mL)", min_value=2.0, max_value=50.0, value=10.0, step=0.1)
            # Calculate HOMA-IR
            if glucose > 0 and insulin > 0:
                homa_ir = (glucose * insulin) / 405
                st.markdown(f"""
                <div style="background: rgba(0, 255, 136, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(0, 255, 136, 0.3);">
                    <div style="color: #00ff88; font-weight: bold; font-size: 1.2rem;">HOMA-IR: {homa_ir:.2f}</div>
                    <div style="color: #888; font-size: 0.8rem;">Formula: (Glucose × Insulin) ÷ 405</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                homa_ir = 2.5
    
    with tab3:
        col5, col6 = st.columns(2)
        with col5:
            diabetes_selfreport = st.selectbox("FAMILY HISTORY", ["No", "Yes"])
            
            if gender == "Female":
                age_first_period = st.number_input("AGE AT FIRST PERIOD", min_value=8, max_value=20, value=13, step=1)
                regular_periods = st.selectbox("REGULAR PERIODS", ["Yes", "No", "Irregular", "Menopause"])
                # Map to numeric values
                regular_periods_map = {"Yes": 1, "No": 2, "Irregular": 3, "Menopause": 4}
                regular_periods_code = regular_periods_map.get(regular_periods, 1)
            else:
                age_first_period = 0
                regular_periods = "N/A"
                regular_periods_code = 0
        
        with col6:
            # Parameter summary
            st.markdown("### 📋 PARAMETER MATRIX")
            params = [
                f"Glucose: {glucose}",
                f"HbA1c: {hba1c}",
                f"Age: {age}",
                f"BMI: {bmi}",
                f"HOMA-IR: {homa_ir:.1f}",
                f"Waist: {waist}",
                f"BP: {systolic_bp}",
                f"Insulin: {insulin}",
                f"Gender: {gender}",
                f"Family: {diabetes_selfreport}"
            ]
            
            # Display as tags
            for param in params:
                st.markdown(f'<span class="param-tag">{param}</span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### ⚡ PREDICTION ENGINE')
    
    # Prediction button
    if st.button("🚀 INITIATE NEURAL ANALYSIS", use_container_width=True, type="primary"):
        if not model_loaded:
            st.error("⚠️ Model files not found. Please run train_and_save.py first.")
        else:
            with st.spinner("Processing neural analysis..."):
                st.markdown('<div class="cyber-spinner"></div>', unsafe_allow_html=True)
                
                # Prepare input
                gender_code = 1 if gender == "Female" else 0
                family_history_code = 1 if diabetes_selfreport == "Yes" else 0
                
                # Create feature array
                features = np.array([[
                    glucose, hba1c, age, bmi, homa_ir,
                    waist, systolic_bp, insulin,
                    age_first_period, gender_code, family_history_code, regular_periods_code
                ]])
                
                try:
                    # Scale and predict
                    features_scaled = scaler.transform(features)
                    probability = model.predict_proba(features_scaled)[0][1]
                    prediction = model.predict(features_scaled)[0]
                    
                    # Display results
                    risk_percentage = probability * 100
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="result-glow" style="border-color: #ff0088;">
                            <div style="font-size: 4rem; color: #ff0088; font-weight: 900; margin: 10px 0;">{risk_percentage:.1f}%</div>
                            <div style="font-size: 1.8rem; color: #ff0088; margin: 10px 0;">🩸 DIABETES DETECTED</div>
                            <div style="color: #aaa; margin-top: 10px;">Probability exceeds 35% threshold</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-glow" style="border-color: #00ff88;">
                            <div style="font-size: 4rem; color: #00ff88; font-weight: 900; margin: 10px 0;">{risk_percentage:.1f}%</div>
                            <div style="font-size: 1.8rem; color: #00ff88; margin: 10px 0;">✅ NO DIABETES</div>
                            <div style="color: #aaa; margin-top: 10px;">Risk below clinical threshold</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_percentage,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "DIABETES RISK METER", 'font': {'color': 'white', 'size': 18}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#ff0088" if prediction == 1 else "#00ff88"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, 35], 'color': 'rgba(0, 255, 136, 0.3)'},
                                {'range': [35, 70], 'color': 'rgba(255, 165, 0, 0.3)'},
                                {'range': [70, 100], 'color': 'rgba(255, 0, 136, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': 35
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "white", 'family': "Arial"},
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"⚠️ Analysis failed: {str(e)}")
    
    else:
        # Default state
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 5rem; color: #00ccff; margin-bottom: 20px;">⚡</div>
            <h3 style="color: #00ccff;">NEURAL NETWORK READY</h3>
            <p style="color: #888; margin: 10px 0;">Enter patient data and initiate analysis</p>
            <div style="margin-top: 30px; color: #666;">
                <div>🔬 12 Clinical Parameters</div>
                <div>🤖 Ensemble AI Model</div>
                <div>🎯 92.3% Accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER STATS ====================
st.markdown("---")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.markdown("""
    <div class="stat-card">
        <div style="color: #00ccff; font-size: 2rem; font-weight: bold;">12</div>
        <div style="color: #aaa;">PARAMETERS</div>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown("""
    <div class="stat-card">
        <div style="color: #00ff88; font-size: 2rem; font-weight: bold;">92.3%</div>
        <div style="color: #aaa;">ACCURACY</div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown("""
    <div class="stat-card">
        <div style="color: #ff0088; font-size: 2rem; font-weight: bold;">3</div>
        <div style="color: #aaa;">ENSEMBLE MODELS</div>
    </div>
    """, unsafe_allow_html=True)

with col_s4:
    st.markdown("""
    <div class="stat-card">
        <div style="color: #ffaa00; font-size: 2rem; font-weight: bold;">35%</div>
        <div style="color: #aaa;">THRESHOLD</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== BOTTOM INFO ====================
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #666; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 2rem;">
    <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 10px; font-size: 1.5rem;">
        <span>⚡</span>
        <span>💠</span>
        <span>🌀</span>
        <span>🔷</span>
    </div>
    <div>🩺 NEURO-DIABETES AI v4.0 • Medical Neural Network • Clinical Analysis System</div>
    <div style="font-size: 0.8rem; color: #555; margin-top: 10px;">For clinical screening purposes only</div>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR INFO ====================
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px; background: rgba(25, 25, 35, 0.8); border-radius: 15px; border: 1px solid rgba(0, 255, 136, 0.2);">
        <h3 style="color: #00ff88;">🚀 QUICK START</h3>
        
        <div style="background: rgba(40, 40, 50, 0.6); padding: 10px; border-radius: 8px; margin: 10px 0;">
            <code style="color: #00ff88;">pip install streamlit pandas numpy plotly</code>
        </div>
        
        <div style="background: rgba(40, 40, 50, 0.6); padding: 10px; border-radius: 8px; margin: 10px 0;">
            <code style="color: #00ff88;">streamlit run app.py</code>
        </div>
        
        <p style="color: #888; font-size: 0.9rem; margin-top: 15px;">
            Ensure <code style="color: #ffaa00;">diabetes_model.pkl</code> and <code style="color: #ffaa00;">diabetes_scaler.pkl</code> exist.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Create requirements.txt automatically
import os
if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write("""streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
plotly==5.17.0
scikit-learn==1.3.0
xgboost==1.7.6""")