# Our scenario continues:

# As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.

# Create a file, explore.py, that contains the following functions for exploring your variables (features & target).

# Write a function, plot_variable_pairs(dataframe) that plots all of the pairwise relationships along with the regression line for each pair.
import matplotlib.pyplot as plt
import seaborn as sns 

import wrangle
import split_scale

X_train, X_test, y_train, y_test = split_scale.split_my_data()

train = pd.merge(X_train, y_train, left_index=True, right_index=True)
test = pd.merge(X_test, y_test, left_index=True, right_index=True)

def plot_variable_pairs(dataframe):
    plot = sns.pairplot(train, x_vars="total_charges", y_vars=["monthly_charges","tenure"])
    return plot 

# Write a function, months_to_years(tenure_months, df) that returns your dataframe with a new feature tenure_years, in complete years as a customer.
def months_to_years(tenure_months, df):
    df["tenure_years"] = round(tenure_months / 12,0)
    return df

# Write a function, plot_categorical_and_continous_vars(categorical_var, continuous_var, df), that outputs 3 different plots for plotting a categorical variable with a continuous variable, e.g. tenure_years with total_charges. For ideas on effective ways to visualize categorical with continuous: https://datavizcatalogue.com/. You can then look into seaborn and matplotlib documentation for ways to create plots.
def plot_categorical_and_continous_vars(categorical_var, continuous_var, df):
    sns.pairplot(train)
    sns.barplot(x="tenure",y="total_charges", data=train) 
    sns.heatmap(train.corr(), annot=True)
    return

