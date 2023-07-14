#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os


def read_data():
    # df = pd.read_parquet(filename)
    # print(df.head())
    options = {
        'client_kwargs': {
            'endpoint_url': 'http://localhost:4566' #S3_ENDPOINT_URL
        }
    }

    df = pd.read_parquet('s3://nyc-duration/yellow_tripdata.parquet', storage_options=options)
        
    return df

def save_data(df):
    # df = pd.read_parquet(filename)
    # print(df.head())
    options = {
        'client_kwargs': {
            'endpoint_url': 'http://localhost:4566' #S3_ENDPOINT_URL
        }
    }

    df.to_parquet(
        's3://nyc-duration/result.parquet',
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
        )


def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data()
    df = prepare_data(df, categorical)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    # print('predicted mean duration:', y_pred.mean())
    print('predicted sum duration:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    # df_result.to_parquet(output_file, engine='pyarrow', index=False)
    save_data(df_result)

if __name__== "__main__":
    main(2022, 2)      