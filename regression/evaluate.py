# Load the tips dataset from either pydataset or seaborn.
from pydataset import data

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

# Write a function, regression_errors(y, yhat), that takes in y and yhat, returns the sum of squared errors (SSE), explained sum of squares (ESS), total sum of squares (TSS), mean squared error (MSE) and root mean squared error (RMSE).
from sklearn.metrics import mean_squared_error

def regression_errors(y, yhat, df):
    SSE = mean_squared_error(y, yhat)*len(df)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE
    MSE = mean_squared_error(y, yhat)
    RMSE = sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE

regression_errors(y,tips.yhat,tips)

# Write a function, baseline_mean_errors(y), that takes in your target, y, computes the SSE, MSE & RMSE when yhat is equal to the mean of all y, and returns the error values (SSE, MSE, and RMSE).
def baseline_mean_errors(y):
    df_baseline = pd.DataFrame(y)
    df_baseline['yhat'] = y.mean()
    df_baseline['residual'] = df_baseline.yhat - df_baseline.iloc[:,0]
    df_baseline['residual^2'] = df_baseline.residual ** 2
    SSE = sum(df_baseline['residual^2'])
    MSE = SSE/len(df_baseline)
    RMSE = sqrt(MSE)
    return SSE, MSE, RMSE

# Write a function, better_than_baseline(SSE), that returns true if your model performs better than the baseline, otherwise false.


# Write a function, model_significance(ols_model), that takes the ols model as input and returns the amount of variance explained in your model, and the value telling you whether the correlation between the model and the tip value are statistically significant.