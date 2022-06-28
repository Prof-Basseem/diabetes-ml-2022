from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
import pickle   # for saving the model

app = Flask(__name__)
model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # get the data from the form
        pregnancies = np.array([float(request.form['pregnancies'])])
        glucose = np.array([float(request.form['glucose'])])
        bloodPressure = np.array([float(request.form['bloodPressure'])])
        skinThickness = np.array([float(request.form['skinThickness'])])
        insulin = np.array([float(request.form['insulin'])])
        imb = np.array([float(request.form['imb'])])
        diabetesPedigreeFunction = np.array([float(request.form['diabetesPedigreeFunction'])])
        age = np.array([float(request.form['age'])])
        # put the data in a dataframe
        data = pd.DataFrame(data={'pregnancies': pregnancies,   'glucose': glucose, 'bloodPressure': bloodPressure, 'skinThickness': skinThickness, 'insulin': insulin, 'imb': imb, 'diabetesPedigreeFunction': diabetesPedigreeFunction, 'age': age})
        # scale the data
        data = scaler.fit_transform(data)
        # make a prediction
        prediction = model.predict(data)
        result = int(prediction[0])
        if prediction == 0:
            result = ['No: diabetes detected']
        else:
            prediction = ['Yes: Diabetes detected']
        # return the prediction
        return render_template('index.html', result=result)
    else:
        return render_template('index.html')

# run the app
if __name__ == '__main__':
    app.run(debug=True, port=9090)
        