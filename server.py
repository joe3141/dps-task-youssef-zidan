import pandas as pd
import numpy as np

import joblib
from flask import Flask, jsonify, request 

from datetime import datetime

af = pd.read_csv("model_files/processed_alcohol_accidents.csv")
af['date'] = pd.to_datetime(af['date'])

model = joblib.load("model_files/model.pkl")
scaler = joblib.load("model_files/scaler.pkl")

def predict(year, month):
	date = str(year) + "-" + str(month) + "-01"
	date = datetime.strptime(date, "%Y-%m-%d")

	prev_months = af[af['date'] < date]
	model_input = prev_months.tail(12)['diff']

	X = model_input.values
	X = np.insert(X, 0, 0.0)
	X = np.array([X])
	X = scaler.transform(X)[:, 1:]

	pred = model.predict(X)

	X = np.insert(X, 0, pred[0]).reshape(1, -1)
	unscaled_pred = scaler.inverse_transform(X)[0][0]

	result = unscaled_pred + prev_months.tail(1)['WERT'].values[0]
	result = int(np.round(result))  # Number of accidents should be an integer

	return result


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_handler():
	request_data = request.get_json()

	year = request_data['year']
	month = request_data['month']

	prediction = predict(year, month)

	return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run()