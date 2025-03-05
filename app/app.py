from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load trained model and imputer
model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
imputer = pickle.load(open('models/imputer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        data = [
            float(request.form['Age']),
            0 if request.form['Gender'] == 'Male' else 1,
            float(request.form['Total_Bilirubin']),
            float(request.form['Direct_Bilirubin']),
            float(request.form['Alkaline_Phosphotase']),
            float(request.form['Alamine_Aminotransferase']),
            float(request.form['Aspartate_Aminotransferase']),
            float(request.form['Total_Protiens']),
            float(request.form['Albumin']),
            float(request.form['Albumin_and_Globulin_Ratio'])
        ]

        # Reshape data and impute if needed
        data = np.array(data).reshape(1, -1)
        data = imputer.transform(data)

        # Predict using the trained model
        prediction = model.predict(data)[0]
        if prediction == 1:
            result = "The prediction indicates that the patient has liver disease."
        else:
            result = "The prediction indicates that the patient does not have liver disease."

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
