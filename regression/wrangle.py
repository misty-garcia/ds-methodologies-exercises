# Throughout the exercises for Regression in Python lessons, you will use the following example scenario: As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.

# The first step will be to acquire and prep the data. Do your work for this exercise in a file named wrangle.py.

# Acquire customer_id, monthly_charges, tenure, and total_charges from telco_churn database for all customers with a 2 year contract.

# # Walk through the steps above using your new dataframe. You may handle the missing values however you feel is appropriate.

# End with a python file wrangle.py that contains the function, wrangle_telco(), that will acquire the data and return a dataframe cleaned with no missing values.

import pandas as pd
import matplotlib as plt
import numpy as np  
import util

def get_data_from_mysql():
    url = util.get_url("telco_churn")
    query = """ 
        SELECT customer_id, monthly_charges, tenure, total_charges
        FROM customers
        JOIN contract_types USING (contract_type_id)
        WHERE contract_type = "Two Year"
        """
    df = pd.read_sql(query, url)
    return df

def clean_data(df):
    df.total_charges = df.total_charges.str.strip().replace("", np.nan).astype(float)
    df = df.dropna()
    return df

def wrangle_telco():
    return clean_data(get_data_from_mysql())
