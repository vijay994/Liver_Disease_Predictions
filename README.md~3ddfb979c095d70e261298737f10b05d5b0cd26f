# 🩺 Liver Disease Prediction - Machine Learning Project

## 📌 Project Objective
The goal of this project is to **predict whether a patient has liver disease or not** based on clinical data using machine learning classification techniques.

---

## 📊 Dataset Information
- **Dataset Name:** Indian Liver Patient Dataset (ILPD)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))
- **Number of Records:** 583
- **Number of Features:** 10 features + 1 target 

Column Name 	              Description
Age	                      Age of the patient
Gender	                  Male/Female
Total_Bilirubin	          Blood Bilirubin Level
Direct_Bilirubin	       Direct Bilirubin Level
Alkaline_Phosphotase	         Enzyme level
Alamine_Aminotransferase	     Enzyme level
Aspartate_Aminotransferase	     Enzyme level
Total_Proteins	Protein level
Albumin	Albumin level
Albumin_and_Globulin_Ratio	  Ratio
Disease (Target)	 1 = Disease, 2 = No Disease

---

## ⚙️ Tech Stack
Component	Tool/Technology
Language	Python
Libraries	Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Flask
ML Models	Logistic Regression, Decision Tree, Random Forest, SVM, KNN
Deployment	Flask (Web App)
---

## 🛠️ Project Workflow
### Step 1 - Data Collection
- Downloaded the dataset from UCI Repository.

### Step 2 - Data Preprocessing
- Handled missing values.
- Encoded categorical feature ("Gender").
- Scaled numeric data if needed.

### Step 3 - Exploratory Data Analysis (EDA)
- Plotted distributions, pair plots, and heatmaps.
- Identified correlations between features.

### Step 4 - Model Building
- Tried multiple classification algorithms:
    - Logistic Regression
    - Decision Tree
    - Random Forest ✅ (Best performing model)
    - KNN
    - SVM

### Step 5 - Model Evaluation
- Compared Accuracy, Precision, Recall, F1-score.
- **Best Model:** Random Forest
- **Accuracy:** 77.85%

### Step 6 - Prediction System (Flask App)
- Built a web form where user can input patient details.
- Backend loads `random_forest_model.pkl` and `imputer.pkl` for preprocessing and prediction.
- Prediction result is shown on a new page.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Liver_Disease_Prediction.git
cd Liver_Disease_Prediction
