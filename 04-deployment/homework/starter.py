#!/usr/bin/env python
# coding: utf-8

import pickle
import sys

from datetime import datetime

import pandas as pd
from flask import Flask, request, jsonify
import mlflow

from prefect import task, flow, get_run_logger
# from prefect import get_run_context


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    return df


def prepare_input_data(df):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    
    return dicts


def predict(dicts, dv, model):
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(y_pred.mean())
    return y_pred


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def save_results(df, y_pred):
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'prediction': y_pred})
    # Save the prediction
    df_result.to_parquet(
        'output_file.parquet',
        engine='pyarrow',
        compression=None,
        index=False
    )


@flow
def ride_duration_prediction():
    logger = get_run_logger()
    year = int(sys.argv[1]) #2022
    month = int(sys.argv[2]) #2

    logger.info('reading data')
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = prepare_input_data(df)

    logger.info('loading the model')
    dv, model = load_model()
    
    logger.info('prediction')
    y_pred = predict(dicts, dv, model)
    
    logger.info('save the results')
    save_results(df, y_pred)
    
if __name__=='__main__':
    ride_duration_prediction()


