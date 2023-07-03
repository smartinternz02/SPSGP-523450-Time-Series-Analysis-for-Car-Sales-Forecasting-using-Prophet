from flask import Flask, render_template, request
import pandas as pd
from prophet import Prophet
import pickle

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    date = request.form['date']

    # Load the model
    with open('forecast.h5', 'rb') as file:
        model = pd.read_pickle(file)

    # Perform the prediction
    future_date = pd.DataFrame({'ds': [date]})
    forecast = model.predict(future_date)
    prediction = forecast['yhat'].values[0]

    return render_template('index.html', prediction=prediction)


if __name__ == '_main_':
    app.run()