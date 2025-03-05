import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Load Data
df = pd.read_csv('D:/Liver_Disease_Predictions/dataset/indian_liver_patient.csv', encoding='ISO-8859-1')

# Clean column names (remove unwanted spaces)
df.columns = df.columns.str.strip()

# Rename target column to a consistent name
if 'Dataset' in df.columns:
    df.rename(columns={'Dataset': 'Liver_Disease'}, inplace=True)
elif 'Result' in df.columns:
    df.rename(columns={'Result': 'Liver_Disease'}, inplace=True)

# Map disease status (1 = Disease, 2 = No Disease)
df['Liver_Disease'] = df['Liver_Disease'].map({1: 1, 2: 0})

# Convert Gender to numeric (0 = Male, 1 = Female)
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
elif 'Gender of the patient' in df.columns:
    df.rename(columns={'Gender of the patient': 'Gender'}, inplace=True)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
numeric_columns = df.drop(columns=['Liver_Disease']).select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Features and Target
X = df.drop('Liver_Disease', axis=1)
y = df['Liver_Disease']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Feature Scaling (needed for models like Logistic Regression, SVM, KNN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to train and evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC()
}

# Track best model
best_model = None
best_accuracy = 0
best_model_name = ''

# Train and Evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}\n")
    
    # Save each model to file
    with open(f'D:/Liver_Disease_Predictions/models/{name.replace(" ", "_").lower()}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Track best performing model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save the best model
with open('D:/Liver_Disease_Predictions/models/best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Save the scaler (needed for predictions later in app.py)
with open('D:/Liver_Disease_Predictions/models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(f"\n Best Model: {best_model_name} with Accuracy: {best_accuracy:.2f}")
print(" All models and scaler saved successfully!")
