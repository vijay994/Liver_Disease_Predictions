import pandas as pd

# Load the dataset
df = pd.read_csv('D:/Liver_Disease_Predictions/dataset/indian_liver_patient.csv', encoding='ISO-8859-1')




# Rename columns for easier handling (optional step, but makes life easy)
df.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkphos_Alkaline_Phosphotase', 
              'Sgpt_Alamine_Aminotransferase', 'Sgot_Aspartate_Aminotransferase', 'Total_Proteins', 
              'Albumin', 'Albumin_and_Globulin_Ratio', 'Liver_Disease']

# Replace 'Male' with 1 and 'Female' with 0 for easier machine learning processing
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Handle missing values in 'Albumin_and_Globulin_Ratio' column
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())

# (Optional) Check if data looks fine after preprocessing
print("Sample Data After Preprocessing:")
print(df.head())

# ✅ Save preprocessed data
# You have 3 options - I am using full path to avoid folder issues
output_path = 'D:/Liver_Disease_Predictions/dataset/preprocessed_liver_patient.csv'

# Make sure the folder 'dataset' already exists in 'Liver_Disease_Predictions'
df.to_csv(output_path, index=False)

print(f"Preprocessed data saved successfully at: {output_path}")
