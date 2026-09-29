import os
import pandas as pd
import numpy as np
import matplotlib

# Turn off interactive screen popups to bypass background framework errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

print("--- Step 1: Initializing Deep Learning Environment ---")

# Locate your baseline Week 1 cleaned data repository
input_path = r"C:\Users\Admin\OneDrive\Desktop\yuva\cleaned_titanic.csv"
output_dir = r"C:\Users\Admin\OneDrive\Desktop\yuva"

df = pd.read_csv(input_path)
print("Cleaned data successfully parsed into memory!")

# Isolate numeric input features and target tracking variables
X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']

# Normalize feature scales for balanced gradient step weights in the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into 80% training matrix and 20% validation split cohort
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

# Step 2: Define and Design the Neural Network Architecture
print("\n--- Constructing Neural Network Model (Multi-Layer Perceptron) ---")
# Architecture: Input Layer (3 features) -> Hidden Layer 1 (8 neurons) -> Hidden Layer 2 (4 neurons) -> Output (1 node)
mlp = MLPClassifier(
    hidden_layer_sizes=(8, 4), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    random_state=42,
    verbose=True # This prints the learning loss optimization across training epochs!
)

# Step 3: Train the Deep Learning Model
print("\n--- Initiating Network Optimization Across Epochs ---")
mlp.fit(X_train, y_train)

# Step 4: Run Model Evaluation Inference
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Neural Network Accuracy Score: {accuracy:.4f}")

# Step 5: Save Network Diagnostics Log File
log_output = f"""=== NEURAL NETWORK ARCHITECTURE & EVALUATION LOG ===
Network Architecture Layout: Input (3) -> Hidden Layer 1 (8 Nodes) -> Hidden Layer 2 (4 Nodes) -> Output Layer (1 Node)
Activation Function: ReLU
Optimization Solver: Adam
Maximum Target Iterations (Epochs): 500

Final Validation Inference Accuracy: {accuracy:.4f}

Detailed Performance Classification Breakdowns:
{classification_report(y_test, y_pred)}
"""

log_file_path = os.path.join(output_dir, "nn_performance_log.txt")
with open(log_file_path, "w") as f:
    f.write(log_output)

# Step 6: Generate Loss Curve Diagram Asset
plt.figure(figsize=(7, 4))
plt.plot(mlp.loss_curve_, color='darkorchid', linewidth=2)
plt.title('Deep Learning Pipeline: Network Loss Convergence Optimization Curve')
plt.xlabel('Training Iteration Steps (Epochs)')
plt.ylabel('Calculated Binary Cross-Entropy Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart_nn_loss_curve.png'))
plt.close()

print(f"\n🎉 Success! 'nn_performance_log.txt' and 'chart_nn_loss_curve.png' saved inside your 'yuva' folder!")
