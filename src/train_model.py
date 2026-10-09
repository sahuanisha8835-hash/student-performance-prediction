from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

from data_preprocessing import load_data, preprocess_data


# Load dataset
data = load_data()

# Preprocess dataset
X, y = preprocess_data(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the Machine Learning model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Find the models folder
models_folder = project_folder / "models"

# Save the trained model
joblib.dump(model, models_folder / "student_model.pkl")

# Save the feature column names
joblib.dump(X.columns.tolist(), models_folder / "feature_columns.pkl")


print("Model trained successfully!")
print("Model saved successfully!")
print("Feature columns saved successfully!")