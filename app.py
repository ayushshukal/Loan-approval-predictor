import streamlit as st
import pickle
import numpy as np

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    min-height: 100vh;
}

h1 { color: #e0e0ff !important; text-align: center; }
h3 { color: #9090cc !important; }

label, .stSelectbox label, .stNumberInput label {
    color: #b0b0dd !important;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c6ef5, #5c5ed6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    width: 100%;
    letter-spacing: 0.06em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88 !important; }

.result-approved {
    background: rgba(56,217,125,0.15);
    border: 1px solid #38d97d;
    border-radius: 14px;
    padding: 22px;
    color: #38d97d;
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
}
.result-rejected {
    background: rgba(255,90,90,0.13);
    border: 1px solid #ff5a5a;
    border-radius: 14px;
    padding: 22px;
    color: #ff5a5a;
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
}
.section-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# ── Header ────────────────────────────────────────
st.markdown("# 🏦 Loan Approval Predictor")
st.markdown("<p style='text-align:center; color:#9090cc;'>Fill in applicant details to check loan eligibility</p>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ `model.pkl` not found. Please run `python model.py` first to train and save the model.")
    st.stop()

st.divider()

# ── Input Form ────────────────────────────────────
st.markdown("### 👤 Personal Details")
col1, col2 = st.columns(2)

with col1:
    gender         = st.selectbox("Gender",        ["Male", "Female"])
    married        = st.selectbox("Married",        ["Yes", "No"])
    dependents     = st.selectbox("Dependents",     ["0", "1", "2", "3+"])

with col2:
    education      = st.selectbox("Education",      ["Graduate", "Not Graduate"])
    self_employed  = st.selectbox("Self Employed",  ["No", "Yes"])
    property_area  = st.selectbox("Property Area",  ["Urban", "Semiurban", "Rural"])

st.markdown("### 💰 Financial Details")
col3, col4 = st.columns(2)

with col3:
    applicant_income   = st.number_input("Applicant Income (₹)",    min_value=0,   value=5000,  step=500)
    loan_amount        = st.number_input("Loan Amount (₹ thousands)", min_value=1,  value=150,   step=10)

with col4:
    coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0,   value=0,     step=500)
    loan_term          = st.number_input("Loan Term (months)",       min_value=12,  value=360,   step=12)

credit_history = st.selectbox("Credit History", ["Good (1) — met all obligations", "Bad (0) — missed payments"])

st.divider()

# ── Predict ───────────────────────────────────────
if st.button("🔍 Check Loan Eligibility"):

    # Encode inputs
    gender_enc        = 1 if gender == "Male" else 0
    married_enc       = 1 if married == "Yes" else 0
    dep_map           = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents_enc    = dep_map[dependents]
    education_enc     = 0 if education == "Graduate" else 1
    self_employed_enc = 1 if self_employed == "Yes" else 0
    area_map          = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_enc      = area_map[property_area]
    credit_enc        = 1 if credit_history.startswith("Good") else 0

    total_income      = applicant_income + coapplicant_income
    total_income_log  = np.log1p(total_income)
    loan_amount_log   = np.log1p(loan_amount)

    features = np.array([[
        gender_enc, married_enc, dependents_enc, education_enc,
        self_employed_enc, applicant_income, coapplicant_income,
        loan_amount, loan_term, credit_enc, property_enc,
        total_income_log, loan_amount_log
    ]])

    prediction  = int(model.predict(features)[0])
    probability = model.predict_proba(features)[0]
    confidence  = round(float(max(probability)) * 100, 2)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-approved">
            ✅ Loan <b>APPROVED</b><br>
            <span style="font-size:0.95rem; font-weight:400;">Confidence: {confidence}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            ❌ Loan <b>REJECTED</b><br>
            <span style="font-size:0.95rem; font-weight:400;">Confidence: {confidence}%</span>
        </div>
        """, unsafe_allow_html=True)

    # Show breakdown
    with st.expander("📊 Probability Breakdown"):
        st.metric("Approved probability", f"{probability[1]*100:.2f}%")
        st.metric("Rejected probability", f"{probability[0]*100:.2f}%")