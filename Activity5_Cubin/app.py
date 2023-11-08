from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model/bank_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    age = float(request.form["age"])
    experience = float(request.form["experience"])
    income = float(request.form["income"])
    family = float(request.form["family"])
    ccavg = float(request.form["ccavg"])
    education = float(request.form["education"])
    mortgage = float(request.form["mortgage"])
    personal_loan = float(request.form["personal_loan"])
    securities_account = float(request.form["securities_account"])
    cd_account = float(request.form["cd_account"])
    online = float(request.form["online"])

    # Make predictions using the loaded model
    prediction = model.predict([[age, experience, income, family, ccavg, education, mortgage, personal_loan,
                                 securities_account, cd_account, online]])

    output = round(prediction[0], 3)

    return render_template('index.html', prediction='Prediction {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
