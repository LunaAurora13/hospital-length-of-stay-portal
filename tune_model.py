import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Dataset & Re-apply Feature Engineering
df = pd.read_csv("hospital_patients_2000.csv")

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

X = df.drop(columns=['Long_Stay'])
y = df['Long_Stay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Grid Search for Hyperparameter Tuning
print("Running Grid Search to find the optimal parameters for 90%+ accuracy...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.8, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest Parameters Found: {grid_search.best_params_}")

# 3. Evaluate Tuned Model
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Tuned Model Accuracy: {accuracy * 100:.2f}% ---")
print(classification_report(y_test, y_pred))

# Save the best model
joblib.dump(best_model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("\nOptimized model successfully saved!")