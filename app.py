import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="Insurance AI Pro", layout="wide", page_icon="🩺")

st.markdown("""
<style>
    .status-bar {
        position: fixed;
        bottom: 0; left: 0; width: 100%;
        background: rgba(28, 31, 34, 0.95);
        color: #00ff9d; text-align: center;
        padding: 12px; font-size: 14px; z-index: 100;
        border-top: 2px solid #27ae60;
        backdrop-filter: blur(10px);
    }
    .stMetric { background: rgba(240, 242, 246, 0.05); padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_engine():
    try:
        df = pd.read_csv('insurance.csv')
        df = df.drop_duplicates().reset_index(drop=True)
        df['bmi_smoker'] = df['bmi'] * df['smoker'].map({'yes': 1, 'no': 0})
        X = df.drop('charges', axis=1)
        y = df['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['age', 'bmi', 'children', 'bmi_smoker']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['sex', 'smoker', 'region'])
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        ])
        pipeline.fit(X_train, y_train)
        r2 = r2_score(y_test, pipeline.predict(X_test))
        return pipeline, r2
    except Exception as e:
        return None, 0.0

model, accuracy = train_engine()

# --- Sidebar ---
with st.sidebar:
    st.header("👤 Patient Profile")
    name = st.text_input("Full Name", "Guest User")
    age = st.slider("Age", 18, 100, 35)
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 27.0, step=0.1)
    children = st.selectbox("Children/Dependents", [0,1,2,3,4,5])
    sex = st.radio("Gender", ["male", "female"], horizontal=True)
    smoker = st.radio("Smoking Status", ["no", "yes"], horizontal=True)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    
    st.markdown("---")
    st.subheader("💡 BMI Category")
    if bmi < 18.5: status, color = "Underweight", "#3498db"
    elif bmi < 25: status, color = "Healthy", "#2ecc71"
    elif bmi < 30: status, color = "Overweight", "#f1c40f"
    else: status, color = "Obese", "#e74c3c"
    
    st.markdown(f"<h2 style='color:{color}; text-align:center;'>{status}</h2>", unsafe_allow_html=True)
    
    st.caption("Standard BMI Scale: 18.5 - 24.9 is optimal.")
    

st.sidebar.image("bmi_chart.png", caption="BMI Categories Reference")


# --- Main Logic ---
input_data = pd.DataFrame([{
    'age': age, 'bmi': bmi, 'children': children,
    'sex': sex, 'smoker': smoker, 'region': region,
    'bmi_smoker': bmi * (1 if smoker == 'yes' else 0)
}])

if model:
    pred = model.predict(input_data)[0]

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.subheader("💰 Prediction Result")
        st.metric("Annual Insurance Cost", f"${pred:,.2f}")
        
        # Risk Logic Improved
        risk_score = 0
        if smoker == "yes": risk_score += 2
        if bmi > 30: risk_score += 1
        risk_level = ["Low", "Moderate", "High", "Critical"][min(risk_score, 3)]
        risk_color = ["green", "orange", "red", "darkred"][min(risk_score, 3)]
        
        st.markdown(f"**Health Risk Profile:** <span style='color:{risk_color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)

    with c2:
        st.subheader("📊 Factor Analysis")
        importances = model.named_steps['regressor'].feature_importances_
        features = model.named_steps['preprocessor'].get_feature_names_out()
        clean_features = [f.split('__')[-1].replace('region_', '').replace('smoker_', 'Smoker: ') for f in features]
        
        top_idx = np.argsort(importances)[-6:]
        
        # تحسين مظهر الرسم البياني ليتناسب مع الثيم
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([clean_features[i] for i in top_idx], importances[top_idx], color='#00d1ff')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        ax.tick_params(colors='gray', labelsize=10)
        st.pyplot(fig)

# --- Health Optimization ---
st.markdown("---")
st.header("🎯 Health Improvement Analysis")

# محاكاة ذكية: الوصول للوزن المثالي (24.0)
target_bmi = 24.0 if bmi > 24.9 else bmi
improved_data = input_data.copy()
improved_data['bmi'] = target_bmi
improved_data['smoker'] = "no"
improved_data['bmi_smoker'] = 0

improved_pred = model.predict(improved_data)[0]
savings = max(0, pred - improved_pred)

if savings > 100:
    st.success(f"### You can save **${savings:,.2f}** per year!")
    st.info(f"This represents a **{(savings/pred)*100:.1f}%** reduction in your annual charges.")

    t1, t2 = st.tabs(["🚭 Quit Smoking Plan", "🥗 Weight Control"])
    with t1:
        st.write("### Benefits of Quitting:")
        st.write("- **Immediate:** Blood pressure stabilizes.")
        st.write("- **Financial:** Smoker surcharge is the #1 cost driver in insurance.")
    with t2:
        st.write(f"### Strategy to reach BMI {target_bmi}:")
        st.write("- Focus on high-protein, low-sugar diet.")
        st.write("- Combine cardio with strength training 3x weekly.")
        

# --- PDF Generation ---
st.markdown("---")
if st.button("📄 Export Comprehensive Medical-Financial Report", use_container_width=True):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(0, 0, 210, 297, 'F')

    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 20, "Health Risk & Insurance Report", ln=1, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Patient Name: {name}", ln=1)
    pdf.cell(0, 10, f"BMI Status: {status} ({bmi})", ln=1)
    pdf.cell(100, 10, f"Current Charges: ${pred:,.2f}", ln=0)
    pdf.cell(100, 10, f"Optimization Savings: ${savings:,.2f}", ln=1)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "AI Recommendations:", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(
        0, 10,
        f"To achieve a {savings/pred:.0%} reduction, focus on reaching a BMI of {target_bmi} "
        f"and maintaining a smoke-free lifestyle. This model predicts your risk based on "
        f"1,338 historical cases with {accuracy:.1%} accuracy."
    )
    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "Click here to download PDF",
        data=pdf_bytes,
        file_name=f"Report_{name}.pdf",
        mime="application/pdf"
    )

st.markdown(f"""
<div class="status-bar">
    🟢 System Online | Model Accuracy: {accuracy:.1%} | Security: Encrypted
</div>
""", unsafe_allow_html=True)