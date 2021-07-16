from flask import Flask, request, jsonify
import joblib
import pandas as pd

application = Flask(__name__)

@application.route('/predict', methods=['POST'])
def predict():
	features_json = request.json
	query = pd.DataFrame(features_json)
	query = query.reindex(columns=col_names)
	
	prediction = list(model.predif(df))
	
	return jsonify({'prediction': str(prediction)})
	
	
@application.route('/status', methods=['GET'])
def status():
	return '<html><body>status ok </body></html>'
	
if __name__ == '__main__':
	model = joblib.load("model.pkl")
	col_names = joblib.load('column_names.pkl')
	
	application.run(debug=True)
