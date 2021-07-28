from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import boto3
from io import BytesIO
from io import StringIO

application = Flask(__name__)


@application.route('/predict', methods=['POST'])
def predict():
    features_json = request.json

    features_df = load_features(features_json['s3_key'], features_json['s3_bucket'])
    prediction = list(model.predif(features_df))

    return jsonify({'prediction': str(prediction)})


@application.route('/status', methods=['GET'])
def status():
    return '<html><body>status ok </body></html>'


def feed_model(features_df, save_features=False):
    model = joblib.load("logreg.joblib")
    col_names = joblib.load('col_names.pkl')

    # https://stackoverflow.com/questions/51663071/sklearn-fit-vs-predict-order-of-columns-matters
    features_df = features_df[col_names]

    print(features_df.columns)

    if save_features:
        features_df.to_csv(path_or_buf='features.csv', sep=',', index=False, mode='w+')
        f = open("actual_cols.txt", "w+")
        j = json.dumps(list(features_df.columns.values))
        f.write(json.dumps(j))
        f.close()

    prediction = list(model.predict(features_df))
    conf = list(model.predict_proba(features_df))
    print(prediction)
    print(conf)
    print(model.classes_)

    # return the probability predictions for last tick
    return conf[-1]


def load_features(s3_key, s3_bucket):
    s3_client = boto3.client('s3')

    b = BytesIO()
    s3_client.download_fileobj(s3_bucket, s3_key, b)
    b.seek(0)

    csv = b.getvalue()
    csv_str = StringIO(str(csv))
    df = pd.read_csv(csv_str)

    return df


if __name__ == '__main__':
    model = joblib.load("model.pkl")
    col_names = joblib.load('column_names.pkl')

    application.run(debug=True)
