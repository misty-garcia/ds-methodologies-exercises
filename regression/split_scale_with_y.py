# Our scenario continues:

# As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.

# Create split_scale.py that will contain the functions that follow. Each scaler function should create the object, fit and transform both train and test. They should return the scaler, train dataframe scaled, test dataframe scaled. Be sure your indices represent the original indices from train/test, as those represent the indices from the original dataframe. Be sure to set a random state where applicable for reproducibility!
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

import wrangle
import util

from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

# split_my_data(X, y, train_pct)
def prepare_for_split():
    df = wrangle.wrangle_telco()
    X = df[["monthly_charges", "tenure"]]
    y = df.total_charges
    return X, y

def split_my_data(X, y, train_pct):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_pct, random_state=123)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)   
    return X_train, X_test, y_train, y_test

def transform_scaler(train, test, scaler):  
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return train_scaled, test_scaled

def inverse_transform_scaler(train_scaled, test_scaled, scaler):  
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
    return train_unscaled, test_unscaled

# standard_scaler()
def standard_scaler():
    X_train, X_test, y_train, y_test = split_my_data()

    X_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_scaled, X_test_scaled = transform_scaler(X_train, X_test, X_scaler)

    y_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(y_train)
    y_train_scaled, y_test_scaled = transform_scaler(y_train, y_test, y_scaler)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler

# scale_inverse()
def scale_inverse(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler):
    X_train_unscaled, X_test_unscaled = inverse_transform_scaler(X_train_scaled, X_test_scaled, X_scaler)

    y_train_unscaled, y_test_unscaled = inverse_transform_scaler(y_train_scaled, y_test_scaled, y_scaler)

    return X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled

# uniform_scaler()
def uniform_scaler():
    X_train, X_test, y_train, y_test = split_my_data()

    X_scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(X_train)
    X_train_scaled, X_test_scaled = transform_scaler(X_train, X_test, scaler)

    y_scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(y_train)
    y_train_scaled, y_test_scaled = transform_scaler(y_train, y_test, scaler)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler

# gaussian_scaler()
def gaussian_scaler():
    X_train, X_test, y_train, y_test = split_my_data()

    X_scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(X_train)
    X_train_scaled, X_test_scaled = transform_scaler(X_train, X_test, scaler)

    y_scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(y_train)
    y_train_scaled, y_test_scaled = transform_scaler(y_train, y_test, scaler)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler

# min_max_scaler()
def min_max_scaler():
    X_train, X_test, y_train, y_test = split_my_data()

    X_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)
    X_train_scaled, X_test_scaled = transform_scaler(X_train, X_test, scaler)

    y_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(y_train)
    y_train_scaled, y_test_scaled = transform_scaler(y_train, y_test, scaler)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler


# iqr_robust_scaler()
def iqr_robust_scaler():
    X_train, X_test, y_train, y_test = split_my_data()

    X_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(X_train)
    X_train_scaled, X_test_scaled = transform_scaler(X_train, X_test, scaler)

    y_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(y_train)
    y_train_scaled, y_test_scaled = transform_scaler(y_train, y_test, scaler)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler


