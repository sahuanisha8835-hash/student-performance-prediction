import pandas as pd
import joblib
from pathlib import Path


# Find project folder
project_folder = Path(__file__).resolve().parent.parent

# Load trained model
model_path = project_folder / "models" / "student_model.pkl"
model = joblib.load(model_path)

# Load feature columns
columns_path = project_folder / "models" / "feature_columns.pkl"
feature_columns = joblib.load(columns_path)


# Get student information
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))
sleep_hours = float(input("Enter sleep hours: "))

internet_access = input("Internet access (Yes/No): ")
parental_support = input("Parental support (High/Medium/Low): ")
extracurricular = input("Extracurricular activity (Yes/No): ")


# Create student data
new_student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "sleep_hours": [sleep_hours],
    "internet_access": [internet_access],
    "parental_support": [parental_support],
    "extracurricular": [extracurricular]
})


# Convert text data into numerical data
new_student_encoded = pd.get_dummies(
    new_student,
    dtype=int
)


# Make sure columns are exactly the same as training data
new_student_encoded = new_student_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)


# Make prediction
prediction = model.predict(new_student_encoded)


# Display result
if prediction[0] == 1:
    print("\nPredicted Result: PASS")
else:
    print("\nPredicted Result: FAIL")