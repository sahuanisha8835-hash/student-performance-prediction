
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Project paths
PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "models" / "student_model.pkl"
COLUMNS_PATH = PROJECT_FOLDER / "models" / "feature_columns.pkl"

# Page title
st.title("🎓 Student Performance Predictor")
st.write(
    "Predict a student's likely result using a Machine Learning model."
)

st.info(
    "This is a learning project. Predictions are estimates, "
    "not guarantees of academic performance."
)

# Load saved model
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns


model, feature_columns = load_model()

# Student input form
st.subheader("Enter Student Details")

with st.form("student_form"):
    study_hours = st.number_input(
        "Daily study hours",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance percentage",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0
    )

    previous_score = st.number_input(
        "Previous score",
        min_value=0.0,
        max_value=100.0,
        value=78.0,
        step=1.0
    )

    sleep_hours = st.number_input(
        "Daily sleep hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    internet_access = st.selectbox(
        "Internet access",
        ["Yes", "No"]
    )

    parental_support = st.selectbox(
        "Parental support",
        ["High", "Medium", "Low"]
    )

    extracurricular = st.selectbox(
        "Participates in extracurricular activities?",
        ["Yes", "No"]
    )

    submitted = st.form_submit_button(
        "Predict Student Result",
        use_container_width=True
    )

# Make prediction
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

    # Convert text categories into numerical columns
    encoded_student = pd.get_dummies(
        new_student,
        dtype=int
    )

    # Match the training feature columns
    encoded_student = encoded_student.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(encoded_student)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Predicted Result: PASS 🎉")
    else:
        st.error("Predicted Result: FAIL")

    st.caption(
        "The prediction is based on the small sample dataset "
        "used to train this model. It should not be used to make "
        "important decisions about a student's future."
    )