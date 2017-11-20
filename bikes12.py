# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:46:34 2017

@author: arun.bhardwaj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import model_selection, preprocessing
from sklearn.metrics import mean_squared_error
import math

hourly = pd.read_csv("C:\\Users\\arun.bhardwaj\\Desktop\\MyOrdner\\DALab\\modeldata.csv")
hourly.head()

#any null value?
hourly.isnull().sum()
hourly.startday =  pd.to_datetime(hourly.startday, errors="coerce")
hourly.dtypes

hourly.head()
df_sorted = hourly.sort_index(level='startday')

#t4 = df_sorted.groupby(['startday']).num_trips.sum().to_frame()
df_sorted = df_sorted.set_index('startday')

#splitting data into 60%, 20% 20% ration
def train_validate_test_split(df):
    train = pd.DataFrame(df.loc['20130805':'20160131'])
    validate = pd.DataFrame(df.loc['20160201':'20160930'])
    test = pd.DataFrame(df.loc['20161001':'20170630'])
    return train, validate, test

train, validate, test = train_validate_test_split(df_sorted)

print(train.shape)
print(validate.shape)
print(test.shape)


df_sorted.head()


#################################
#fit with ARIMA model with original data

float_num_trips = df_sorted.num_trips.apply(lambda x : float(x))

#float_num_trips_1 = df_sorted.diff_num_trips_1.apply(lambda x : float(x))
float_num_trips.dropna(inplace=True)

from statsmodels.tsa.arima_model import ARIMA 
model = ARIMA(float_num_trips, order=(1, 1, 1))
results_ARIMA = model.fit(disp=-1)

print(results_ARIMA.summary())

##################################
#creating dataframe with mean
model_df = df_sorted
model_df['num_trips_90mean'] = model_df.num_trips.rolling(window=90, min_periods=7, center=False).mean()

############################
#ARIMA test on 90 days mean data
float_num_trips_1 = model_df.num_trips_90mean.apply(lambda x : float(x))
float_num_trips_1.dropna(inplace=True)
print(float_num_trips_1)
model1 = ARIMA(float_num_trips_1, order=(0, 0, 0))
results_ARIMA = model1.fit(disp=-1)
print(results_ARIMA.summary())

##########################
#SARIMAX
import statsmodels.tsa.statespace.sarimax as sm

mod =  sm.SARIMAX(model_df.num_trips_90mean, trend='n', order=(0,1,0), seasonal_order=(0,0,1,4))
results = mod.fit()
print(results.summary())






