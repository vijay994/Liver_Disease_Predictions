import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv('D:/Liver_Disease_Predictions/dataset/indian_liver_patient.csv', encoding='ISO-8859-1')

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Handle column name for target label (works for both 'Dataset' and 'Result')
if 'Dataset' in df.columns:
    df.rename(columns={'Dataset': 'Liver_Disease'}, inplace=True)
elif 'Result' in df.columns:
    df.rename(columns={'Result': 'Liver_Disease'}, inplace=True)

# Map disease status (1 = Disease Present, 2 = No Disease)
df['Liver_Disease'] = df['Liver_Disease'].map({1: 'Disease Present', 2: 'No Disease'})

# Basic Information
print("\nClass Distribution (Disease Present = 1, No Disease = 2):")
print(df['Liver_Disease'].value_counts())

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values Count:")
print(df.isnull().sum())

# --- Plot 1: Age Distribution ---
plt.figure(figsize=(8,5))
sns.histplot(df['Age of the patient'], kde=True, color='skyblue')

plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# --- Plot 2: Gender Distribution ---
plt.figure(figsize=(6,4))
sns.countplot(x='Gender of the patient', data=df, palette='Set2')

plt.title('Gender Count')
plt.grid(axis='y')
plt.show()

# --- Plot 3: Target Class Distribution (Disease Present/Absent) ---
plt.figure(figsize=(6,4))
sns.countplot(x='Liver_Disease', data=df, palette='coolwarm')
plt.title('Liver Disease Distribution')
plt.grid(axis='y')
plt.show()

# --- Plot 4: Albumin vs Globulin Ratio ---
plt.figure(figsize=(7,5))
sns.boxplot(x='Liver_Disease', y='A/G Ratio Albumin and Globulin Ratio', data=df, palette='pastel')
plt.title('Albumin and Globulin Ratio by Disease Status')
plt.show()

# --- Plot 5: Correlation Heatmap ---
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# --- Plot 6: Bilirubin Levels by Disease Status ---
plt.figure(figsize=(10,5))
sns.boxplot(x='Liver_Disease', y='Total Bilirubin', data=df, palette='muted')
plt.title('Total Bilirubin by Disease Status')
plt.show()

# --- Extra Analysis Tip ---
print("\nGender-wise Distribution for Disease Presence:")
print(df.groupby(['Gender of the patient', 'Liver_Disease']).size())
