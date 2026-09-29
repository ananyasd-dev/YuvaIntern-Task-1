import os
import sys
import subprocess

# Automatically verify and install scikit-learn if it is missing
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
except ModuleNotFoundError:
    print("--- Scikit-learn missing. Installing library now... ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import numpy as np

print("--- Step 1: Initializing Supervised Learning Setup ---")

# Locate Week 1 cleaned data repository
input_path = r"C:\Users\Admin\OneDrive\Desktop\yuva\cleaned_titanic.csv"
output_dir = r"C:\Users\Admin\OneDrive\Desktop\yuva"

df = pd.read_csv(input_path)
print("Dataset parsed successfully!")

# Isolate predictive features (X) and target flag (y)
X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']

# Split data into training cohort (80%) and test confirmation matrix (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Step 2: Initialize Logistic Regression Classifier
print("\n--- Training Supervised Classification Engine ---")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 3: Run Validation Routines
cv_scores = cross_val_score(model, X_train, y_train, cv=3)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Cross-Validation Stability Metric: {cv_scores.mean():.4f}")
print(f"Test Subset Predictive Accuracy: {accuracy:.4f}")

# Step 4: Write Model Evaluation Report Text Log File
log_output = f"""=== SUPERVISED MODEL EVALUATION DATA LOG ===
Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}
Holdout Split Final Test Accuracy: {accuracy:.4f}

Classification Performance Breakdown:
{classification_report(y_test, y_pred)}
"""

log_file_path = os.path.join(output_dir, "model_performance_log.txt")
with open(log_file_path, "w") as f:
    f.write(log_output)

print(f"\n🎉 Success! 'model_performance_log.txt' has been generated inside your 'yuva' folder!")
