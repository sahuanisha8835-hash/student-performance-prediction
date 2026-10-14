
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Student AI | Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM DASHBOARD DESIGN
# ==========================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111a32 100%);
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #0a0f1e;
        border-right: 1px solid #253252;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    p, label {
        color: #cbd5e1;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #17233e, #1c2850);
        border: 1px solid #33446b;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stForm"] {
        background-color: #121c33;
        border: 1px solid #293858;
        padding: 24px;
        border-radius: 16px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        border: 1px solid #93c5fd;
        color: white;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. PROJECT PATHS
# ==========================================
PROJECT_FOLDER = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_FOLDER / "models" / "student_model.pkl"
COLUMNS_PATH = PROJECT_FOLDER / "models" / "feature_columns.pkl"
DATA_PATH = PROJECT_FOLDER / "data" / "student_data.csv"

# ==========================================
# 4. LOAD MACHINE LEARNING MODEL
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns


try:
    model, feature_columns = load_model()
except FileNotFoundError:
    st.error(
        "Model files were not found. Please train your model first "
        "by running src/train_model.py."
    )
    st.stop()

# ==========================================
# 5. LOAD STUDENT DATASET
# ==========================================
student_data = None

if DATA_PATH.exists():
    student_data = pd.read_csv(DATA_PATH)
else:
    st.warning(
        "Student dataset not found. Check data/student_data.csv."
    )

# ==========================================
# 6. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🎓 Student AI")
    st.caption("Performance Analytics Platform")
    st.divider()

    st.subheader("Navigation")
    st.write("🏠 Dashboard")
    st.write("📊 Student Analytics")
    st.write("🧠 ML Prediction")

    st.divider()
    st.subheader("About Project")

    st.write(
        "An educational Machine Learning project that estimates "
        "a student's likely result from academic and lifestyle details."
    )

    st.divider()
    st.caption(
        "Built with Python, Pandas, Scikit-learn and Streamlit"
    )

# ==========================================
# 7. DASHBOARD HEADER
# ==========================================
st.title("🎓 Student Performance Analytics")

st.write(
    "An AI-powered dashboard to explore student information "
    "and estimate academic outcomes."
)

st.info(
    "Learning project only: predictions are estimates, "
    "not guarantees of academic performance."
)

# ==========================================
# 8. DASHBOARD METRIC CARDS
# ==========================================
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        label="🤖 Machine Learning",
        value="Active",
        delta="Model loaded"
    )

with metric2:
    st.metric(
        label="📊 Input Features",
        value=len(feature_columns),
        delta="Model features"
    )

with metric3:
    st.metric(
        label="🎯 Prediction Type",
        value="Pass / Fail",
        delta="Classification"
    )

st.divider()



# ==========================================
# 9. STUDENT DATA ANALYTICS
# ==========================================
st.header("📊 Student Performance Analytics")

if student_data is not None:
    # Count students in each result category
    result_counts = (
        student_data["result"]
        .astype(str)
        .str.strip()
        .str.title()
        .value_counts()
        .reindex(["Pass", "Fail"], fill_value=0)
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Pass vs Fail Distribution")

        pass_count = int(result_counts["Pass"])
        fail_count = int(result_counts["Fail"])

        metric_pass, metric_fail = st.columns(2)

        metric_pass.metric("✅ Passed", pass_count)
        metric_fail.metric("❌ Failed", fail_count)

        st.bar_chart(
            result_counts,
            y_label="Number of Students",
            color="#3B82F6"
        )

    with chart_col2:
        st.subheader("Study Hours vs Previous Score")

        st.scatter_chart(
            student_data,
            x="study_hours",
            y="previous_score",
            x_label="Daily Study Hours",
            y_label="Previous Score",
            color="#A78BFA"
        )

    st.subheader("📋 Student Dataset Preview")

    st.dataframe(
        student_data,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("Analytics are unavailable until the dataset is added.")

# ==========================================
# 10. STUDENT INPUT FORM
# ==========================================
st.header("🧠 Student Result Prediction")

st.write(
    "Enter the student's details and run the prediction."
)

with st.form("student_form"):
    st.subheader("Academic Information")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "📚 Daily study hours",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5
        )

        attendance = st.number_input(
            "📅 Attendance percentage",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=1.0
        )

    with col2:
        previous_score = st.number_input(
            "📝 Previous score",
            min_value=0.0,
            max_value=100.0,
            value=78.0,
            step=1.0
        )

        sleep_hours = st.number_input(
            "😴 Daily sleep hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

    st.divider()
    st.subheader("Student Background")

    col3, col4, col5 = st.columns(3)

    with col3:
        internet_access = st.selectbox(
            "🌐 Internet access",
            ["Yes", "No"]
        )

    with col4:
        parental_support = st.selectbox(
            "👨‍👩‍👧 Parental support",
            ["High", "Medium", "Low"]
        )

    with col5:
        extracurricular = st.selectbox(
            "⚽ Extracurricular activities",
            ["Yes", "No"]
        )

    submitted = st.form_submit_button(
        "🚀 Predict Student Result",
        use_container_width=True
    )

# ==========================================
# 11. MAKE PREDICTION
# ==========================================
if submitted:
    new_student = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_score": [previous_score],
        "sleep_hours": [sleep_hours],
        "internet_access": [internet_access],
        "parental_support": [parental_support],
        "extracurricular": [extracurricular]
    })

    # Convert text categories into numerical columns.
    encoded_student = pd.get_dummies(
        new_student,
        dtype=int
    )

    # Match the feature columns used during training.
    encoded_student = encoded_student.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(encoded_student)[0]

    # ======================================
    # 12. DISPLAY PREDICTION RESULT
    # ======================================
    st.divider()
    st.header("📈 Prediction Results")

    result_col, details_col = st.columns([1, 2])

    with result_col:
        if prediction == 1:
            st.success("### PASS 🎉")
            st.write("Predicted outcome: Pass")
        else:
            st.error("### FAIL")
            st.write("Predicted outcome: Fail")

    with details_col:
        st.subheader("Student Summary")

        st.write(f"📚 Study hours: **{study_hours} hours/day**")
        st.write(f"📅 Attendance: **{attendance:.0f}%**")
        st.write(f"📝 Previous score: **{previous_score:.0f}/100**")
        st.write(f"😴 Sleep: **{sleep_hours} hours/day**")
        st.write(f"🌐 Internet access: **{internet_access}**")
        st.write(f"👨‍👩‍👧 Parental support: **{parental_support}**")
        st.write(f"⚽ Extracurricular: **{extracurricular}**")

    st.warning(
        "This prediction comes from a very small sample dataset. "
        "It is a learning demonstration, not a reliable assessment "
        "of a real student's academic future."
    )

# ==========================================
# 13. FOOTER
# ==========================================
st.divider()

st.caption(
    "Student AI | Machine Learning Project | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
