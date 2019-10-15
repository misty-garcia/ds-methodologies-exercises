# Our scenario continues:
# As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.
import pandas as pd 
import numpy as np 
import split_scale
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm

# Write a function, select_kbest_freg_unscaled() that takes X_train, y_train and k as input (X_train and y_train should not be scaled!) and returns a list of the top k features.
df = split_scale.prepare_telco_for_split()
train, test = split_scale.split_my_data(df, .80)

X_train = train.drop(columns = "total_charges")
y_train = train["total_charges"]

def select_kbest_freg_unscaled(X_train, y_train, k):
    f_selector = SelectKBest(f_regression, k=k).fit(X_train, y_train)
    f_support = f_selector.get_support()
    f_feature = X_train.loc[:,f_support].columns.tolist()
    return f_feature

select_kbest_freg_unscaled(X_train, y_train, 2)

# Write a function, select_kbest_freg_scaled() that takes X_train, y_train (scaled) and k as input and returns a list of the top k features.
train_scaled, train_scaled, scaler = split_scale.standard_scaler(train, test)

X_train_scaled = train_scaled.drop(columns = "total_charges")
y_train_scaled = train_scaled["total_charges"]

def select_kbest_freg_scaled(X_train, y_train, k):
    f_selector = SelectKBest(f_regression, k=k).fit(X_train, y_train)
    f_support = f_selector.get_support()
    f_feature = X_train.loc[:,f_support].columns.tolist()
    return f_feature

select_kbest_freg_scaled(X_train_scaled, y_train_scaled, 2)

# Write a function, ols_backware_elimination() that takes X_train and y_train (scaled) as input and returns selected features based on the ols backwards elimination method.
def ols_backware_elimination(X_train, y_train):
    ols_model = sm.OLS(y_train, X_train)
    fit = ols_model.fit()

    cols = list(X_train.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = X_train[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y_train,X_1).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    return cols

ols_backware_elimination(X_train, y_train)

# Write a function, lasso_cv_coef() that takes X_train and y_train as input and returns the coefficients for each feature, along with a plot of the features and their weights.

# Write 3 functions, the first computes the number of optimum features (n) using rfe, the second takes n as input and returns the top n features, and the third takes the list of the top n features as input and returns a new X_train and X_test dataframe with those top features , recursive_feature_elimination() that computes the optimum number of features (n) and returns the top n features.