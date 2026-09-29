import os
import pandas as pd
import numpy as np
import matplotlib

# Turn off interactive windows to bypass PySide6 background errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

print("--- Step 1: Initializing EDA Environment ---")

# Directly locate your Week 1 file paths
input_path = r"C:\Users\Admin\OneDrive\Desktop\yuva\cleaned_titanic.csv"
output_dir = r"C:\Users\Admin\OneDrive\Desktop\yuva"

# Load the spreadsheet
df = pd.read_csv(input_path)
print("Cleaned data from Week 1 loaded successfully!")

# Step 2: Print Basic Summary Statistics to the Terminal window
print("\n--- Descriptive Summary Analysis ---")
print(df.describe())

print("\n--- Categorical Proportions (Survival Breakdown) ---")
print(df['Survived'].value_counts(normalize=True))

# CHART 1: Distribution of Passenger Age (Histogram using built-in matplotlib)
plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=10, color='teal', edgecolor='black', alpha=0.7)
plt.title('Exploratory Analysis: Demographics - Age Distribution Profile')
plt.xlabel('Passenger Age (Years)')
plt.ylabel('Frequency Count')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart_age_distribution.png'))
plt.close()

# CHART 2: Survival Status Based on Passenger Class (Bar Plot using group calculations)
plt.figure(figsize=(8, 5))
class_survival = df.groupby('Pclass')['Survived'].mean()
colors = ['#4c72b0', '#55a868', '#c44e52']

plt.bar(class_survival.index.astype(str), class_survival.values, color=colors, edgecolor='black', alpha=0.8)
plt.title('Exploratory Analysis: Comparative Insights - Survival Density Across Ticket Classes')
plt.xlabel('Ticket Placement Rank (1st, 2nd, 3rd Class)')
plt.ylabel('Calculated Probability Value')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'chart_survival_by_class.png'))
plt.close()

print(f"\n🎉 Success! Two diagnostic visualization plots have been saved inside your 'yuva' folder!")
