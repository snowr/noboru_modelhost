from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	features_json = request.json
	query = pd.DataFrame(features_json)
	query = query.reindex(columns=col_names)
	
	prediction = list(model.predif(df))
	
	return jsonify({'prediction': str(prediction)})
	
	
@app.route('/status', methods=['GET'])
def status()
	return jsonify({'status': 'ok'})
	
if __name__ == '__main__':
	model = joblib.load("model.pkl")
	col_names = joblib.load('column_names.pkl')
	
	app.run(debug=True)