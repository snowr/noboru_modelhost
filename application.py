from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import boto3
import logging
from io import BytesIO
from io import StringIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)

application = Flask(__name__)


@application.route('/predict', methods=['POST'])
def predict():
    logger.info(f'json: {request.json}')
    features_json = request.json

    features_df = load_features(features_json['s3_key'], features_json['s3_bucket'])
    conf = feed_model(features_df)

    return jsonify({'conf_0': conf[0], 'conf_1': conf[1]})


@application.route('/status', methods=['GET'])
def status():
    return '<html><body>status ok </body></html>'


# returns array[conf_zero, conf_1]
def feed_model(features_df, save_features=False):
    model = joblib.load("logreg.joblib")
    col_names = joblib.load('col_names.pkl')

    logger.info(f'features_df before trim {len(features_df.columns.values)}')
    # https://stackoverflow.com/questions/51663071/sklearn-fit-vs-predict-order-of-columns-matters
    features_df = features_df.drop(['TradingDateTimestamp', 'TradingDate', 'stock_price',
                                    'Unnamed: 0',
                                    'call_volume',
                                    'call_volume_10_delta_to_mean',
                                    'call_volume_10_stddev',
                                    'call_volume_15_delta_to_mean',
                                    'call_volume_15_stddev',
                                    'call_volume_20_delta_to_mean',
                                    'call_volume_20_stddev',
                                    'call_volume_25_delta_to_mean',
                                    'call_volume_25_stddev',
                                    'call_volume_30_delta_to_mean',
                                    'call_volume_30_stddev',
                                    'call_volume_45_delta_to_mean',
                                    'call_volume_45_stddev',
                                    'call_volume_5_delta_to_mean',
                                    'call_volume_5_stddev',
                                    'call_volume_60_delta_to_mean',
                                    'call_volume_60_stddev',
                                    'put_volume',
                                    'put_volume_10_delta_to_mean',
                                    'put_volume_10_stddev',
                                    'put_volume_15_delta_to_mean',
                                    'put_volume_15_stddev',
                                    'put_volume_20_delta_to_mean',
                                    'put_volume_20_stddev',
                                    'put_volume_25_delta_to_mean',
                                    'put_volume_25_stddev',
                                    'put_volume_30_delta_to_mean',
                                    'put_volume_30_stddev',
                                    'put_volume_45_delta_to_mean',
                                    'put_volume_45_stddev',
                                    'put_volume_5_delta_to_mean',
                                    'put_volume_5_stddev',
                                    'put_volume_60_delta_to_mean',
                                    'put_volume_60_stddev',
                                    'stock_volume',
                                    'stock_volume_10_delta_to_mean',
                                    'stock_volume_10_stddev',
                                    'stock_volume_15_delta_to_mean',
                                    'stock_volume_15_stddev',
                                    'stock_volume_20_delta_to_mean',
                                    'stock_volume_20_stddev',
                                    'stock_volume_25_delta_to_mean',
                                    'stock_volume_25_stddev',
                                    'stock_volume_30_delta_to_mean',
                                    'stock_volume_30_stddev',
                                    'stock_volume_45_delta_to_mean',
                                    'stock_volume_45_stddev',
                                    'stock_volume_5_delta_to_mean',
                                    'stock_volume_5_stddev',
                                    'stock_volume_60_delta_to_mean',
                                    'stock_volume_60_stddev',
                                    ], axis=1, errors='ignore')
    logger.info(f'features_df after trim {len(features_df.columns.values)}')
    logger.info(f'model col length: {(len(col_names))}')
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
    # model = joblib.load("model.pkl")
    # col_names = joblib.load('column_names.pkl')

    application.run(debug=True)
