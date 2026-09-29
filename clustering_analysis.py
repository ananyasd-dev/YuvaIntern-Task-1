import os
import sys
import subprocess

# Automatically install scikit-learn if it is missing from the environment
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    print("--- Scikit-learn missing. Installing library now... ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib

# Deactivate interactive window components to prevent background toolkit crashes
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

print("--- Step 1: Initializing Clustering Environment ---")

# Locate Week 1 cleaned database file path
input_path = r"C:\Users\Admin\OneDrive\Desktop\yuva\cleaned_titanic.csv"
output_dir = r"C:\Users\Admin\OneDrive\Desktop\yuva"

df = pd.read_csv(input_path)
print("Cleaned data loaded successfully!")

# Select features for unsupervised clustering analysis
features = ['Age', 'Fare']
X = df[features]

# Standardize features (Mean=0, Variance=1) for balanced K-Means performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply K-Means Clustering Algorithm (Targeting K=3 distinct groups)
print("\n--- Executing K-Means Clustering Algorithmic Logic (K=3) ---")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 3: Profile and display cluster properties in the terminal window
print("\n--- In-Depth Cluster Profiling Metrics ---")
cluster_profile = df.groupby('Cluster')[features].mean()
print(cluster_profile)

# CHART: Generate Scatter Plot highlighting the segmented customer groups
plt.figure(figsize=(8, 6))
colors = ['#4c72b0', '#55a868', '#c44e52']

for cluster_id in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster_id]
    plt.scatter(cluster_data['Age'], cluster_data['Fare'], 
                label=f'Cluster {cluster_id}', c=colors[cluster_id], 
                s=100, edgecolor='black', alpha=0.8)

plt.title('Unsupervised Learning: Passenger Segmentation Profile (K-Means)', fontsize=14, pad=15)
plt.xlabel('Passenger Age (Years)', fontsize=11)
plt.ylabel('Ticket Fare Value ($)', fontsize=11)
plt.legend(title="Segment Profiles")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save image file to your folder layout path
plt.savefig(os.path.join(output_dir, 'cluster_segmentation_chart.png'))
plt.close()

print(f"\n🎉 Success! The unsupervised learning segmentation chart has been saved to your 'yuva' folder!")
