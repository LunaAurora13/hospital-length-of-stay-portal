import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib

# 1. Load Dataset and Display Preview
print("Loading dataset...")
df = pd.read_csv("hospital_patients_2000.csv")

# Print dataset shape and first 5 rows (just like your preferred output)
print(f"Dataset Shape: {df.shape}")
print(df.head())
print("\n" + "="*50 + "\n")

# 2. Advanced Feature Engineering
if 'Systolic_BP' in df.columns and 'Diastolic_BP' in df.columns:
    df['Pulse_Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']

binary_cols = ['Diabetes', 'Hypertension', 'Previous_Hospitalization']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)

df['Comorbidity_Score'] = df['Diabetes'] + df['Hypertension'] + df['Previous_Hospitalization']

if 'Patient_ID' in df.columns:
    df = df.drop(columns=['Patient_ID'])
if 'Length_of_Stay_Days' in df.columns:
    df = df.drop(columns=['Length_of_Stay_Days'])

categorical_cols = ['Gender', 'Disease', 'Severity', 'Admission_Type']
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Define Features (X) and Target (y)
X = df.drop(columns=['Long_Stay'])
y = df['Long_Stay']

# 3. Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Gradient Boosting Classifier (Best Performing Setup)
print("Training Gradient Boosting model with engineered features...")
model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Gradient Boosting Results ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the Winning Model, Scaler, and Encoders
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("\nWinning model successfully saved!")