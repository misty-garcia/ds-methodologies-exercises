# Load the tips dataset from either pydataset or seaborn.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from math import sqrt
from pydataset import data
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

tips = data("tips")

# Fit a linear regression model (ordinary least squares) and compute yhat, predictions of tip using total_bill. You may follow these steps to do that:

# import the method from statsmodels: from statsmodels.formula.api import ols
from statsmodels.formula.api import ols

# fit the model to your data, where x = total_bill and y = tip: regr = ols('y ~ x', data=df).fit()
x = tips.total_bill
y = tips.tip
regr = ols('y ~ x', data=tips).fit()

# compute yhat, the predictions of tip using total_bill: df['yhat'] = regr.predict(df.x)
tips["yhat"] = regr.predict(pd.DataFrame(x))

# Create a file evaluate.py that contains the following functions.

# Write a function, plot_residuals(x, y, dataframe) that takes the feature, the target, and the dataframe as input and returns a residual plot. (hint: seaborn has an easy way to do this!)
def plot_residuals(x, y, dataframe):
    sns.residplot(x, y, data = dataframe)
    return

plot_residuals(x,y,tips)

# Write a function, regression_errors(y, yhat), that takes in y and yhat, returns the sum of squared errors (SSE), explained sum of squares (ESS), total sum of squares (TSS), mean squared error (MSE) and root mean squared error (RMSE).
def regression_errors(y, yhat):
    SSE = mean_squared_error(y, yhat)*len(y)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE
    MSE = mean_squared_error(y, yhat)
    RMSE = sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE

regression_errors(y,tips.yhat)

# Write a function, baseline_mean_errors(y), that takes in your target, y, computes the SSE, MSE & RMSE when yhat is equal to the mean of all y, and returns the error values (SSE, MSE, and RMSE).
def baseline_mean_errors(y):
    df_baseline = pd.DataFrame(y)
    df_baseline['yhat'] = y.mean()
    MSE = mean_squared_error(df_baseline.iloc[:,0], df_baseline.yhat)
    SSE = MSE*len(df_baseline)
    RMSE = sqrt(MSE)
    return SSE, MSE, RMSE

baseline_mean_errors(tips.tip)

# Write a function, better_than_baseline(SSE), that returns true if your model performs better than the baseline, otherwise false.
def better_than_baseline(SSE, SSE_base):
    return SSE < SSE_base

regression = regression_errors(y,tips.yhat)
baseline = baseline_mean_errors(tips.tip)
better_than_baseline(regression[0], baseline[0])

# Write a function, model_significance(ols_model), that takes the ols model as input and returns the amount of variance explained in your model, and the value telling you whether the correlation between the model and the tip value are statistically significant.
def model_significance(ols_model):
    r2 = ols_model.rsquared
    f_pval = ols_model.f_pvalue
    return r2, f_pval

model_significance(regr)
