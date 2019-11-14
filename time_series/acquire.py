import pandas as pd
import requests

# Using the code from the lesson as a guide, create a dataframe named items that has all of the data for items.
def get_items():
    base_url = "http://python.zach.lol"
    response = requests.get(base_url + "/api/v1/items")
    items = pd.DataFrame(response.json()["payload"]["items"])

    response = requests.get(base_url + "/api/v1/items?page=2")
    items = pd.concat([items,pd.DataFrame(response.json()["payload"]["items"])], ignore_index=True)
    
    response = requests.get(base_url + "/api/v1/items?page=3")
    items = pd.concat([items,pd.DataFrame(response.json()["payload"]["items"])], ignore_index=True)
    return items

# Do the same thing, but for stores.
def get_stores():
    base_url = "http://python.zach.lol"
    response = requests.get(base_url + "/api/v1/stores")
    stores = pd.DataFrame(response.json()["payload"]["stores"])
    return stores

# Extract the data for sales. There are a lot of pages of data here, so your code will need to be a little more complex. Your code should continue fetching data from the next page until all of the data is extracted.
def get_sales():
    base_url = "http://python.zach.lol"
    response = requests.get(base_url + "/api/v1/sales")

    maxpage = response.json()["payload"]["max_page"] + 1
    sales = pd.DataFrame()

    for page in range(1,maxpage):
        response = requests.get(base_url + "/api/v1/sales?page={}".format(page))
        sale_page = pd.DataFrame(response.json()["payload"]["sales"])
        sales = pd.concat([sales,sale_page])
    return sales

# Save the data in your files to local csv files so that it will be faster to access in the future.
def df_to_csv(df):
    return df.to_csv('/Users/mists/codeup-data-science/ds-methodologies-exercises/time_series/sales.csv')

def get_sales_csv():
    sales = pd.read_csv("sales.csv")
    sales.drop(columns="Unnamed: 0", inplace=True)
    return sales

# Combine the data from your three separate dataframes into one large dataframe.
def combine_dfs():
    stores = get_stores()
    items = get_items()
    sales = get_sales_csv()
    df = sales.merge(stores, left_on="store", right_on="store_id")
    df = df.merge(items, left_on="item", right_on="item_id", how="left")
    return df

# Acquire the Open Power Systems Data for Germany, which has been rapidly expanding its renewable energy production in recent years. The data set includes country-wide totals of electricity consumption, wind power production, and solar power production for 2006-2017. You can get the data here: https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.cs
def get_OPS():
    return pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')

# Make sure all the work that you have done above is reproducible. That is, you should put the code above into separate functions in the acquire.py file and be able to re-run the functions and get the same data.