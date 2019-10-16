# Our scenario continues:
# As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.
import util
import wrangle
import split_scale

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LassoCV, LinearRegression

# Write a function, select_kbest_freg_unscaled() that takes X_train, y_train and k as input (X_train and y_train should not be scaled!) and returns a list of the top k features.

df = pd.read_sql("""
    SELECT payment_type_id, monthly_charges, total_charges, tenure
    FROM customers
    """, util.get_url("telco_churn"))

df = wrangle.clean_data(df)
df.info()

# df = split_scale.prepare_telco_for_split()
train, test = split_scale.split_my_data(df, .80)

X_train = train.drop(columns = "total_charges")
y_train = train["total_charges"]

def select_kbest_freg_unscaled(X_train, y_train, k):
    f_selector = SelectKBest(f_regression, k=k).fit(X_train, y_train)
    f_support = f_selector.get_support()
    f_feature = X_train.loc[:,f_support].columns.tolist()
    return f_feature

select_kbest_freg_unscaled(X_train, y_train, 1)

# Write a function, select_kbest_freg_scaled() that takes X_train, y_train (scaled) and k as input and returns a list of the top k features.
train_scaled, test_scaled, scaler = split_scale.standard_scaler(train, test)

X_train_scaled = train_scaled.drop(columns = "total_charges")
y_train_scaled = train_scaled["total_charges"]
X_test_scaled = test_scaled.drop(columns = "total_charges")
y_test_scaled = test_scaled["total_charges"]

def select_kbest_freg_scaled(X_train, y_train, k):
    f_selector = SelectKBest(f_regression, k=k).fit(X_train, y_train)
    f_support = f_selector.get_support()
    f_feature = X_train.loc[:,f_support].columns.tolist()
    return f_feature

select_kbest_freg_scaled(X_train_scaled, y_train_scaled, 1)

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

ols_backware_elimination(X_train_scaled, y_train_scaled)

# Write a function, lasso_cv_coef() that takes X_train and y_train as input and returns the coefficients for each feature, along with a plot of the features and their weights.
def lasso_cv_coef(X_train, y_train):
    reg = LassoCV(cv=5)
    reg.fit(X_train, y_train)

    coef = pd.Series(reg.coef_, index = X_train.columns)
    imp_coef = coef.sort_values()

    plot = imp_coef.plot(kind = "barh")
    return coef, plot

lasso_cv_coef(X_train_scaled, y_train_scaled)

# Write 3 functions, the first computes the number of optimum features (n) using rfe, the second takes n as input and returns the top n features, and the third takes the list of the top n features as input and returns a new X_train and X_test dataframe with those top features , recursive_feature_elimination() that computes the optimum number of features (n) and returns the top n features.
def num_optimum_features(X_train, y_train, X_test, y_test):
    model = LinearRegression() 
    rfe = RFE(model, 2)
    X_rfe = rfe.fit_transform(X_train,y_train)  
    model.fit(X_rfe,y_train)

    number_of_features_list=np.arange(1,3)
    high_score=0
    number_of_features=0           
    score_list =[]

    for n in range(len(number_of_features_list)):
        model = LinearRegression()
        rfe = RFE(model,number_of_features_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            number_of_features = number_of_features_list[n]
    return number_of_features

def top_features(n):
    cols = list(X_train_scaled.columns)

    model = LinearRegression()
    rfe = RFE(model, n)
    X_rfe = rfe.fit_transform(X_train_scaled,y_train_scaled)  

    model.fit(X_rfe,y_train_scaled)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index

    return selected_features_rfe

def n_to_X_train_and_test(features):
    features = list(features)
    new_X_train = X_train_scaled[features]
    new_X_test = X_test_scaled[features]
    return new_X_train, new_X_test

def recursive_feature_elimination(X_train, y_train, X_test, y_test):
    return top_features(num_optimum_features(X_train, y_train, X_test, y_test))

#tests
n = num_optimum_features(X_train_scaled, y_train_scaled, X_test_scaled,y_test_scaled)

selected_features = top_features(n)

n_to_X_train_and_test(selected_features)

recursive_feature_elimination(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
