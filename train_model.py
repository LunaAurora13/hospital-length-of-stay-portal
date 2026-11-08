import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv("hospital_patients_2000.csv")
print(f"Dataset Shape: {df.shape}")
print(df.head())

# 2. Data Preprocessing
if 'Patient_ID' in df.columns:
    df = df.drop(columns=['Patient_ID'])
if 'Length_of_Stay_Days' in df.columns:
    df = df.drop(columns=['Length_of_Stay_Days'])

# Encode categorical columns
categorical_cols = ['Gender', 'Disease', 'Severity', 'Admission_Type', 'Diabetes', 'Hypertension', 'Previous_Hospitalization']
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Define Features (X) and Target (y)
X = df.drop(columns=['Long_Stay'])
y = df['Long_Stay']

# 3. Split Dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Logistic Regression Model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the Model, Scaler, and Encoders
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("\nModel, scaler, and encoders successfully saved!")