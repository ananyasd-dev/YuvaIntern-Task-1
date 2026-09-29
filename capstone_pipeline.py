import os
import pandas as pd
import numpy as np
import matplotlib

# Deactivate interactive window components to prevent runtime server/UI crashes
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

print("--- Step 1: Initializing Integrative Capstone Environment ---")

# Locate baseline cleaned database paths
input_path = r"C:\Users\Admin\OneDrive\Desktop\yuva\cleaned_titanic.csv"
output_dir = r"C:\Users\Admin\OneDrive\Desktop\yuva"

df = pd.read_csv(input_path)
print("Data pipeline baseline loaded successfully!")

# ==========================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\n--- Phase 2: Processing Capstone Visualization Graphics ---")
plt.figure(figsize=(6, 4))
df.groupby('Pclass')['Survived'].mean().plot(kind='bar', color=['#4c72b0', '#55a868', '#c44e52'], edgecolor='black')
plt.title('Capstone Metrics: Survival Probability Matrix by Ticket Class')
plt.xlabel('Ticket Placement Rank (1st, 2nd, 3rd Class)')
plt.ylabel('Survival Ratio Value')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'capstone_eda_chart.png'))
plt.close()

# ==========================================
# PHASE 3: UNSUPERVISED LEARNING (CLUSTERING)
# ==========================================
print("\n--- Phase 3: Executing Unsupervised Clustering Arrays ---")
cluster_features = ['Age', 'Fare']
scaler = StandardScaler()
X_scaled_clus = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_Label'] = kmeans.fit_predict(X_scaled_clus)

# ==========================================
# PHASE 4 & 5: SUPERVISED MODELING & DEEP LEARNING
# ==========================================
print("\n--- Phase 4 & 5: Training Comparative Supervised Models ---")
X_model = df[['Pclass', 'Age', 'Fare']]
y_model = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.20, random_state=42)

# Normalizing inputs for consistent scale weights
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model A: Supervised Logistic Regression Classifier
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)

# Model B: Multi-Layer Perceptron Neural Network Architecture
mlp_nn = MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', max_iter=500, random_state=42)
mlp_nn.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_nn.predict(X_test_scaled)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

# ==========================================
# PHASE 6: LOG PRODUCTION & REPORT VALIDATION
# ==========================================
print("\n--- Phase 6: Compiling Final Performance Artifact Log ---")
capstone_log = f"""=======================================================
INTEGRATIVE DATA SCIENCE CAPSTONE PIPELINE LOG REPORT
=======================================================
Total Processed Cohort Volume: {len(df)} Records
Features Selected: Pclass, Age, Fare
Target Classification Element: Survived (Binary)

[UNSUPERVISED METRICS]
K-Means Structural Cluster Counts Configured: K=3 Distinct Archetypes

[SUPERVISED COMPARATIVE ANALYSIS]
- Linear Logistic Regression Validation Accuracy: {acc_log:.4f}
- Artificial Neural Network (MLP) Validation Accuracy: {acc_mlp:.4f}

Detailed Artificial Neural Network Performance Breakdown:
{classification_report(y_test, y_pred_mlp)}
"""

final_log_path = os.path.join(output_dir, "capstone_final_evaluation.txt")
with open(final_log_path, "w") as f:
    f.write(capstone_log)

print(f"\n🎉 Success! 'capstone_eda_chart.png' and 'capstone_final_evaluation.txt' saved in your 'yuva' folder!")
