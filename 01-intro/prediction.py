import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error


def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


df = pd.read_parquet('data/yellow_tripdata_2022-01.parquet')

# print(df.head())
print(df.shape)

# print(df.columns)


df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

# Statistic 
# print(df.duration.std())

# Dropping outliers
df = df[(df.duration >= 1) & (df.duration <= 60)]
# print(df_clean.shape[0]/df.shape[0])

# One-hot encoding
categorical = ['PULocationID', 'DOLocationID']
# numerical = ['trip_distance']

df[categorical] = df[categorical].astype(str)

train_dicts = df[categorical].to_dict(orient='records')

dv = DictVectorizer()
# print(len(dv.feature_names_))

X_train = dv.fit_transform(train_dicts)

target = 'duration'
y_train = df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

# print(mean_squared_error(y_train, y_pred, squared=False))

# validate the model
# df_train = read_dataframe('data/yellow_tripdata_2022-01.parquet')
df_val = read_dataframe('data/yellow_tripdata_2022-02.parquet')

print(len(df_val))

val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_val = df_val[target].values

y_pred = lr.predict(X_val)

print(mean_squared_error(y_val, y_pred, squared=False))


with open('models/lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)


