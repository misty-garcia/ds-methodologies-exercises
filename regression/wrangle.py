import pandas as pd
import matplotlib as plt
import numpy as np  
import util

# Throughout the exercises for Regression in Python lessons, you will use the following example scenario: As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.

# The first step will be to acquire and prep the data. Do your work for this exercise in a file named wrangle.py.

# Acquire customer_id, monthly_charges, tenure, and total_charges from telco_churn database for all customers with a 2 year contract.

url = util.get_url("telco_churn")
pd.read_sql("SHOW TABLES", url)

query = """ 
    SELECT customer_id, monthly_charges, tenure, total_charges, contract_type
    FROM customers
    JOIN contract_types USING (contract_type_id)
    WHERE contract_type = "Two Year"
    """
df1 = pd.read_sql(query, url)
df1.head()

# Walk through the steps above using your new dataframe. You may handle the missing values however you feel is appropriate.
df1.info()
df1.shape

df1.sort_values(by="total_charges")

df1.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df1.isnull().sum()

df1.replace(np.nan, 0, inplace=True)
df1.total_charges = df1.total_charges.astype("float")

df1 = df1.dropna()

# End with a python file wrangle.py that contains the function, wrangle_telco(), that will acquire the data and return a dataframe cleaned with no missing values.

def wrangle_telco(df):
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    return df
