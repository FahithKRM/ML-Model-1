from flask import Flask, render_template, request, redirect
import pickle

import numpy as np

app = Flask(__name__)

with open("linear_regression_model.pkl", 'rb') as file :
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    mid_semseter_marks = request.form['mid']

    input_data = [[float(mid_semseter_marks)]]
    reshaped_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(reshaped_data)

    return render_template('index.html', prediction=prediction[0])


if __name__ == '__main__' :
    app.run(debug=True)